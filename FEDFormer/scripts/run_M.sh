export CUDA_VISIBLE_DEVICES=1

#cd ..

for model in FEDformer Autoformer Informer
do

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Informer-data.csv \
  --task_id ETTm1 \
  --model $model \
  --data ETTm1 \
  --features m \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 1 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 97 \
  --dec_in 97 \
  --c_out 97 \
  --des 'Exp' \
  --d_model 512 \
  --itr 3 \
done

