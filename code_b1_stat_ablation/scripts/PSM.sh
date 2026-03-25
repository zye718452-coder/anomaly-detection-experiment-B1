export CUDA_VISIBLE_DEVICES=0

python main.py \
  --anormly_ratio 1 \
  --num_epochs 3 \
  --batch_size 256 \
  --mode train \
  --dataset PSM \
  --data_path dataset/PSM \
  --input_c 25 \
  --output_c 25 \
  --use_b1 true \
  --ma_window 5 \
  --alpha_base 1.2 \
  --alpha_range 0.3 \
  --alpha_hidden -1 \
  "$@"

python main.py \
  --anormly_ratio 1 \
  --num_epochs 10 \
  --batch_size 256 \
  --mode test \
  --dataset PSM \
  --data_path dataset/PSM \
  --input_c 25 \
  --output_c 25 \
  --pretrained_model 20 \
  --use_b1 true \
  --ma_window 5 \
  --alpha_base 1.2 \
  --alpha_range 0.3 \
  --alpha_hidden -1 \
  "$@"