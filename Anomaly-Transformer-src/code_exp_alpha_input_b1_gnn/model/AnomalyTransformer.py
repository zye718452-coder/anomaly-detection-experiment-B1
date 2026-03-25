import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, mask, sigma


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        series_list = []
        prior_list = []
        sigma_list = []

        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class MovingAvgBackground(nn.Module):
    """
    Channel-wise moving average background extractor.
    Input:  x [B, L, C]
    Output: bg [B, L, C]
    """
    def __init__(self, kernel_size=5):
        super(MovingAvgBackground, self).__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd."
        self.kernel_size = kernel_size
        self.pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        x_t = x.transpose(1, 2)  # [B, C, L]

        if self.pad > 0:
            left = x_t[:, :, 0:1].repeat(1, 1, self.pad)
            right = x_t[:, :, -1:].repeat(1, 1, self.pad)
            x_pad = torch.cat([left, x_t, right], dim=-1)
        else:
            x_pad = x_t

        bg = self.avg(x_pad)     # [B, C, L]
        bg = bg.transpose(1, 2)  # [B, L, C]
        return bg


class GraphAlphaGenerator(nn.Module):
    """
    G2:
    data-adaptive graph + global prior graph fusion for unified enhancement.

    Input:
        residual: [B, L, C]
    Output:
        alpha:    [B, 1, C]
        res_stat: [B, C, 4]
        A:        [B, C, C]
    """
    def __init__(self,
                 num_channels,
                 stat_dim=4,
                 node_emb_dim=16,
                 hidden_dim=16,
                 graph_feat_dim=16,
                 alpha_base=1.2,
                 alpha_scale=0.3,
                 graph_lambda=0.3,
                 fusion_beta=0.1,
                 prior_beta=0.2,
                 eps=1e-6):
        super(GraphAlphaGenerator, self).__init__()
        self.num_channels = num_channels
        self.stat_dim = stat_dim
        self.node_emb_dim = node_emb_dim
        self.hidden_dim = hidden_dim
        self.graph_feat_dim = graph_feat_dim
        self.alpha_base = alpha_base
        self.alpha_scale = alpha_scale
        self.graph_lambda = graph_lambda
        self.fusion_beta = fusion_beta
        self.prior_beta = prior_beta
        self.eps = eps

        # global node embedding prior
        self.node_embedding = nn.Parameter(torch.randn(num_channels, node_emb_dim) * 0.02)

        # local branch
        self.local_fc1 = nn.Linear(stat_dim, hidden_dim)
        self.local_fc2 = nn.Linear(hidden_dim, hidden_dim)

        # graph branch
        self.graph_fc = nn.Linear(stat_dim, hidden_dim)

        # data-adaptive graph constructor
        self.graph_query = nn.Linear(stat_dim, graph_feat_dim)
        self.graph_key = nn.Linear(stat_dim, graph_feat_dim)

        # output head
        self.fusion_fc = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, 1)

        self.hidden_norm = nn.LayerNorm(hidden_dim)

    def extract_res_stat(self, residual):
        """
        residual: [B, L, C]
        Output:
            res_stat: [B, C, 4]
                0: mean(abs(res))
                1: max(abs(res))
                2: std(abs(res))
                3: energy = mean(res^2)
        """
        abs_res = residual.abs()

        res_mean = abs_res.mean(dim=1)                # [B, C]
        res_max = abs_res.max(dim=1)[0]               # [B, C]
        res_std = abs_res.std(dim=1, unbiased=False)  # [B, C]
        res_energy = (residual ** 2).mean(dim=1)      # [B, C]

        res_stat = torch.stack(
            [res_mean, res_max, res_std, res_energy],
            dim=-1
        )  # [B, C, 4]

        return res_stat

    def build_adaptive_adjacency(self, h0):
        """
        h0: [B, C, stat_dim]
        Output:
            A: [B, C, C]
        """
        # sample-dependent graph
        q = self.graph_query(h0)                      # [B, C, G]
        k = self.graph_key(h0)                        # [B, C, G]
        sim_data = torch.matmul(q, k.transpose(1, 2))  # [B, C, C]
        sim_data = F.relu(sim_data)

        # global prior graph
        sim_prior = torch.matmul(
            self.node_embedding,
            self.node_embedding.transpose(0, 1)
        )  # [C, C]
        sim_prior = F.relu(sim_prior)
        sim_prior = sim_prior.unsqueeze(0)            # [1, C, C]

        # self-loop
        eye = torch.eye(self.num_channels, device=h0.device, dtype=h0.dtype).unsqueeze(0)

        sim = sim_data + self.prior_beta * sim_prior + eye
        A = torch.softmax(sim, dim=-1)                # [B, C, C]
        return A

    def forward(self, residual):
        """
        residual: [B, L, C]
        """
        B, L, C = residual.shape
        assert C == self.num_channels, "Channel size mismatch in GraphAlphaGenerator."

        # Step 1: node feature init
        h0 = self.extract_res_stat(residual)          # [B, C, 4]

        # Step 2: local branch
        h_local = F.relu(self.local_fc1(h0))          # [B, C, H]
        h_local = self.local_fc2(h_local)             # [B, C, H]

        # Step 3: adaptive graph branch
        A = self.build_adaptive_adjacency(h0)         # [B, C, C]
        h_graph_in = self.graph_fc(h0)                # [B, C, H]
        msg = torch.matmul(A, h_graph_in)             # [B, C, H]
        h_graph = h_graph_in + self.graph_lambda * F.relu(msg)

        # Step 4: local + graph fusion
        h = h_local + self.fusion_beta * h_graph
        h = self.hidden_norm(h)
        h = F.relu(self.fusion_fc(h))

        # Step 5: alpha in [alpha_base - alpha_scale, alpha_base + alpha_scale]
        delta = self.alpha_scale * torch.tanh(self.out_fc(h))   # [B, C, 1]
        alpha = self.alpha_base + delta                         # [B, C, 1]
        alpha = alpha.transpose(1, 2)                           # [B, 1, C]

        return alpha, h0, A


class AnomalyTransformer(nn.Module):
    def __init__(self,
                 win_size,
                 enc_in,
                 c_out,
                 d_model=512,
                 n_heads=8,
                 e_layers=3,
                 d_ff=512,
                 dropout=0.0,
                 activation='gelu',
                 output_attention=True,
                 bg_kernel_size=5,
                 graph_emb_dim=16,
                 graph_hidden_dim=16,
                 graph_feat_dim=16,
                 alpha_base=1.2,
                 alpha_scale=0.3,
                 graph_lambda=0.3,
                 fusion_beta=0.1,
                 prior_beta=0.2):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # unified front-end enhancement
        self.bg_extractor = MovingAvgBackground(kernel_size=bg_kernel_size)
        self.alpha_generator = GraphAlphaGenerator(
            num_channels=enc_in,
            stat_dim=4,
            node_emb_dim=graph_emb_dim,
            hidden_dim=graph_hidden_dim,
            graph_feat_dim=graph_feat_dim,
            alpha_base=alpha_base,
            alpha_scale=alpha_scale,
            graph_lambda=graph_lambda,
            fusion_beta=fusion_beta,
            prior_beta=prior_beta
        )

        # cached stats for logging
        self.last_alpha = None
        self.last_adj = None
        self.last_res_stat = None

        # original backbone unchanged
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            win_size,
                            False,
                            attention_dropout=dropout,
                            output_attention=output_attention
                        ),
                        d_model,
                        n_heads
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def front_end_enhance(self, x):
        """
        x: [B, L, C]
        """
        bg = self.bg_extractor(x)                  # [B, L, C]
        residual = x - bg                          # [B, L, C]
        alpha, res_stat, A = self.alpha_generator(residual)

        x_enhanced = x + alpha * residual

        # cache for debugging/logging
        self.last_alpha = alpha.squeeze(1).detach()   # [B, C]
        self.last_adj = A.detach()                    # [B, C, C]
        self.last_res_stat = res_stat.detach()        # [B, C, 4]

        aux_info = {
            "bg": bg,
            "residual": residual,
            "alpha": alpha,
            "res_stat": res_stat,
            "adjacency": A
        }
        return x_enhanced, aux_info

    def forward(self, x):
        # keep original 4-return interface for solver compatibility
        x, _ = self.front_end_enhance(x)

        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
            return enc_out