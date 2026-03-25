export CUDA_VISIBLE_DEVICES=0

python main.py \
  --anormly_ratio 1 \
  --num_epochs 3 \
  --batch_size 256 \
  --mode train \
  --dataset MSL \
  --data_path dataset/MSL \
  --input_c 55 \
  --output_c 55 \
  --use_score_fusion 1 \
  --lambda_evi 0.005 \
  --score_fusion_ma 5 \
  --score_fusion_eps 1e-4

python main.py \
  --anormly_ratio 1 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode test \
  --dataset MSL \
  --data_path dataset/MSL \
  --input_c 55 \
  --output_c 55 \
  --pretrained_model 20 \
  --use_score_fusion 1 \
  --lambda_evi 0.005 \
  --score_fusion_ma 5 \
  --score_fusion_eps 1e-4