export CUDA_VISIBLE_DEVICES=0

python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode train --dataset SMD --data_path dataset/SMD --input_c 38 --output_c 38 --use_b1 true --ma_window 5 --alpha_base 1.2 --alpha_range 0.3 --alpha_hidden -1

python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset SMD --data_path dataset/SMD --input_c 38 --output_c 38 --pretrained_model 20 --use_b1 true --ma_window 5 --alpha_base 1.2 --alpha_range 0.3 --alpha_hidden -1