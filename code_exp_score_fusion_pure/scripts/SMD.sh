export CUDA_VISIBLE_DEVICES=0

python main.py \
  --anormly_ratio 0.5 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode train \
  --dataset SMD \
  --data_path dataset/SMD \
  --input_c 38 \
  --output_c 38 \
  --use_score_fusion 1 \
  --lambda_evi 0.005 \
  --score_fusion_ma 5 \
  --score_fusion_eps 1e-4

python main.py \
  --anormly_ratio 0.5 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode test \
  --dataset SMD \
  --data_path dataset/SMD \
  --input_c 38 \
  --output_c 38 \
  --pretrained_model 20 \
  --use_score_fusion 1 \
  --lambda_evi 0.005 \
  --score_fusion_ma 5 \
  --score_fusion_eps 1e-4