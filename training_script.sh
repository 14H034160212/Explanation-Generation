## Convert the LLaMA-7B to LLaMA-7B huggingface model
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ../../LLaMA/7B \
    --model_size 7B \
    --output_dir llama_7B_hf

## Convert the LLaMA-13B to LLaMA-13B huggingface model
python transformers/src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir ../../LLaMA/13B \
    --model_size 13B \
    --output_dir llama_13B_hf

## Fine-tuning the LLaMA-7B and replicate the Alpaca-7B model
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./alpaca_data.json \
   --bf16 True \
   --output_dir qiming_alpaca \
   --num_train_epochs 3 \
   --per_device_train_batch_size 4 \
   --per_device_eval_batch_size 4 \
   --gradient_accumulation_steps 8 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the LLaMA-13B and replicate the Alpaca-13B model
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2023 train.py \
   --model_name_or_path llama_13B_hf \
   --data_path ./alpaca_data.json \
   --bf16 True \
   --output_dir qiming_alpaca_13B \
   --num_train_epochs 3 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the LLaMA-7B using new PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_generator.json \
   --bf16 True \
   --output_dir qiming_llama_7B_Cardiff_Sydney_merged_generator \
   --model_max_length 1024 \
   --num_train_epochs 3 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the LLaMA-13B using new PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_13B_hf \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_generator.json \
   --bf16 True \
   --output_dir qiming_llama_13B_Cardiff_Sydney_merged_generator \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 1e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the LLaMA-7B using new PeerWise dataset for explanation verifier way 1
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_verifier_way_1.json \
   --bf16 True \
   --output_dir qiming_llama_7B_Cardiff_Sydney_merged_verifier_way_1 \
   --num_train_epochs 3 \
   --model_max_length 1024 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True


## Fine-tuning the LLaMA-7B using new PeerWise dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_verifier_way_2.json \
   --bf16 True \
   --output_dir qiming_llama_7B_Cardiff_Sydney_merged_verifier_way_2 \
   --num_train_epochs 3 \
   --model_max_length 1024 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the Alpaca-7B using new merged PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path qiming_alpaca_7B \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_generator.json \
   --bf16 True \
   --output_dir qiming_alpaca_7B_Cardiff_Sydney_merged_generator \
   --model_max_length 1024 \
   --num_train_epochs 20 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True


## Fine-tuning the Alpaca-7B using Cardiff only PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=2025 train.py \
   --model_name_or_path qiming_alpaca_7B \
   --data_path ./Paul_new_data/Cardiff_generator_train.json \
   --bf16 True \
   --output_dir qiming_alpaca_7B_Cardiff_generator \
   --model_max_length 1024 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 5 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the LLaMA-7B using Cardiff only PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port=2025 train.py \
   --model_name_or_path llama_7B_hf/llama-7b \
   --data_path ./Paul_new_data/Cardiff_generator_train.json \
   --bf16 True \
   --output_dir LLaMA_7B_Cardiff_generator \
   --model_max_length 1024 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 5 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the LLaMA-13B using Cardiff only PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2026 train.py \
   --model_name_or_path llama_13B_hf \
   --data_path ./Paul_new_data/Cardiff_generator_train.json \
   --bf16 True \
   --output_dir LLaMA_13B_Cardiff_generator \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 5 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the Vicuna-13B using UK Medical Year 1 only avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./PeerWiseData/Medicine/UK_medical_year1_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir vicuna_13B_UK_medical_year1_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the Vicuna-13B using UK Medical Year 2 only avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./PeerWiseData/Medicine/UK_medical_year2_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir vicuna_13B_UK_medical_year2_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the Vicuna-13B using Auckland law only avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./PeerWiseData/Law/Auckland_law_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir vicuna_13B_Auckland_law_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True


## Fine-tuning the Vicuna-13B using Cardiff only avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./Paul_new_data/Cardiff_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir vicuna_13B_Cardiff_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the Vicuna-13B using Sydney all avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 torchrun --nproc_per_node=7 --master_port=2026 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./Paul_new_data/Sydney_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir vicuna_13B_Sydney_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the Vicuna-13B using merged all avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir vicuna_13B_merged_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the llama-2-7B using merged all avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-7B \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir llama_2_7B_merged_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the llama-2-13B using merged all avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10_update.json \
   --bf16 True \
   --output_dir llama_2_13B_merged_all_generator_avg_3_lenexp_10_update \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True


## Fine-tuning the llama-2-13B using merged all avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama_13B_chat_hf \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir llama_2_13B_chat_merged_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the stabilityai/StableBeluga-13B using merged all avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=1,3,4,5,6,7 torchrun --nproc_per_node=6 --master_port=2027 train.py \
   --model_name_or_path stabilityai/StableBeluga-13B \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir StableBeluga-13B_merged_all_generator_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the llama-2-13B using Sydney avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/Sydney_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir llama_2_13B_sydney_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True


## Fine-tuning the llama-2-13B using Cardiff avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/Cardiff_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir llama_2_13B_cardiff_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the llama-2-13B using Auckland law avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/Auckland_law_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir llama_2_13B_auckland_law_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True


## Fine-tuning the llama-2-13B using uk medical year 1 avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/UK_medical_year1_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir llama_2_13B_uk_medical_year1_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the llama-2-13B using uk medical year 2 avg >=3 and explanation length >=10 PeerWise dataset for explanation generator
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/UK_medical_year2_all_generator_train_avg_3_lenexp_10.json \
   --bf16 True \
   --output_dir llama_2_13B_uk_medical_year2_avg_3_lenexp_10 \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the llama-2-13B using new PeerWise dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./Paul_new_data/evaluator_Train_all.json \
   --bf16 True \
   --output_dir llama_2_13B_merged_all_evaluator \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True


## Fine-tuning the llama-2-13B using new PeerWise dataset Sydney for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./Paul_new_data/Sydney/evaluator_Train_sydney.json \
   --bf16 True \
   --output_dir llama_2_13B_sydney_evaluator \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True


## Fine-tuning the llama-2-13B using new PeerWise dataset Cardiff for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./Paul_new_data/Cardiff/evaluator_Train_cardiff.json \
   --bf16 True \
   --output_dir llama_2_13B_cardiff_evaluator \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True


## Fine-tuning the llama-2-13B using new PeerWise dataset Auckland Law for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./PeerWiseData/Law/evaluator_Train_auckland_law.json \
   --bf16 True \
   --output_dir llama_2_13B_auckland_law_evaluator \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True

## Fine-tuning the llama-2-13B using new PeerWise dataset Medicine year 1 for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./PeerWiseData/Medicine/evaluator_Train_uk_medical_year1.json \
   --bf16 True \
   --output_dir llama_2_13B_uk_medical_year1_evaluator \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True


## Fine-tuning the llama-2-13B using new PeerWise dataset Medicine year 2 for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2026 train.py \
   --model_name_or_path llama-2/llama-2-13B \
   --data_path ./PeerWiseData/Medicine/evaluator_Train_uk_medical_year2.json \
   --bf16 True \
   --output_dir llama_2_13B_uk_medical_year2_evaluator \
   --model_max_length 512 \
   --num_train_epochs 5 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True \
   --gradient_checkpointing True


## Fine-tuning the Alpaca-7B using new PeerWise dataset for explanation verifier way 1
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path qiming_alpaca_7B \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_verifier_way_1.json \
   --bf16 True \
   --output_dir qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_1 \
   --num_train_epochs 20 \
   --model_max_length 1024 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the Alpaca-7B using new PeerWise dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master_port=2024 train.py \
   --model_name_or_path qiming_alpaca_7B \
   --data_path ./Paul_new_data/Cardiff_Sydney_merged_verifier_way_2.json \
   --bf16 True \
   --output_dir qiming_alpaca_7B_Cardiff_Sydney_merged_verifier_way_2 \
   --num_train_epochs 20 \
   --model_max_length 1024 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True


## Fine-tuning the Vicuna-13B using new PeerWise Sydney and Sydney additional dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2024 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./Paul_new_data/Sydney_merged_verifier_way_2.json \
   --bf16 True \
   --output_dir qiming_vicuna_13B_Sydney_merged_verifier_way_2 \
   --num_train_epochs 5 \
   --model_max_length 512 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True


## Fine-tuning the Vicuna-13B using new PeerWise Cardiff dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2024 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./Paul_new_data/Cardiff_merged_verifier_way_2.json \
   --bf16 True \
   --output_dir qiming_vicuna_13B_Cardiff_merged_verifier_way_2 \
   --num_train_epochs 5 \
   --model_max_length 512 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the Vicuna-13B using new PeerWise Auckland law dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2024 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./PeerWiseData/Law/Auckland_law_merged_verifier_way_2.json \
   --bf16 True \
   --output_dir qiming_vicuna_13B_Auckland_law_merged_verifier_way_2 \
   --num_train_epochs 5 \
   --model_max_length 512 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the Vicuna-13B using new PeerWise UK Medicine year1 dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2024 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./PeerWiseData/Medicine/UK_medicine_year1_merged_verifier_way_2.json \
   --bf16 True \
   --output_dir qiming_vicuna_13B_UK_medicine_year1_merged_verifier_way_2 \
   --num_train_epochs 5 \
   --model_max_length 512 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

## Fine-tuning the Vicuna-13B using new PeerWise UK Medicine year2 dataset for explanation verifier way 2
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port=2024 train.py \
   --model_name_or_path vicuna-13b \
   --data_path ./PeerWiseData/Medicine/UK_medicine_year2_merged_verifier_way_2.json \
   --bf16 True \
   --output_dir qiming_vicuna_13B_UK_medicine_year2_merged_verifier_way_2 \
   --num_train_epochs 5 \
   --model_max_length 512 \
   --per_device_train_batch_size 1 \
   --per_device_eval_batch_size 1 \
   --gradient_accumulation_steps 16 \
   --evaluation_strategy "no" \
   --save_strategy "steps" \
   --save_steps 2000 \
   --save_total_limit 1 \
   --learning_rate 2e-5 \
   --weight_decay 0. \
   --warmup_ratio 0.03 \
   --lr_scheduler_type "cosine" \
   --logging_steps 1 \
   --fsdp "full_shard auto_wrap" \
   --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
   --tf32 True

python gpt-4_script.py \
    --data_path ./Paul_new_data/Sydney/round2/Sydney_all_generator_test_avg_3_lenexp_10_round2_random_sample_100.json \
    --output_path ./Paul_new_data/Sydney/round2/Sydney_gpt4_round2_random_100.json \
    --excel_output_path ./Paul_new_data/Sydney/round2/Sydney_gpt4_round2_random_100.json \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_script.py \
    --data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_random_100.json \
    --output_path ./Paul_new_data/Cardiff/Cardiff_gpt4_random_100.json \
    --excel_output_path ./Paul_new_data/Cardiff/Cardiff_gpt4_random_100.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_script.py \
    --data_path ./PeerWiseData/Law/Auckland_law_vicuna_13b_random_100.json \
    --output_path ./PeerWiseData/Law/Auckland_law_gpt4_random_100.json \
    --excel_output_path ./PeerWiseData/Law/Auckland_law_gpt4_random_100.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_script.py \
    --data_path ./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_random_100.json \
    --output_path ./PeerWiseData/Medicine/Medicine_year1_gpt4_random_100.json \
    --excel_output_path ./PeerWiseData/Medicine/Medicine_year1_gpt4_random_100.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_script.py \
    --data_path ./PeerWiseData/Medicine/Medicine_year2_vicuna_13b_random_100.json \
    --output_path ./PeerWiseData/Medicine/Medicine_year2_gpt4_random_100.json \
    --excel_output_path ./PeerWiseData/Medicine/Medicine_year2_gpt4_random_100.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_script.py \
    --data_path ./Paul_new_data/Sydney/Sydney_gpt-4_random_100.json \
    --output_path ./Paul_new_data/Sydney/Sydney_gpt-4_random_100_correct.json \
    --excel_output_path ./Paul_new_data/Sydney/Sydney_gpt-4_random_100_correct.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY









python gpt-4_script.py \
    --data_path ./Paul_new_data/Cardiff/Cardiff_vicuna_13b_random_100.json \
    --output_path ./Paul_new_data/Cardiff/Cardiff_gpt35_random_100.json \
    --excel_output_path ./Paul_new_data/Cardiff/Cardiff_gpt35_random_100.xlsx \
    --model_name gpt-3.5-turbo \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_script.py \
    --data_path ./PeerWiseData/Law/Auckland_law_vicuna_13b_random_100.json \
    --output_path ./PeerWiseData/Law/Auckland_law_gpt35_random_100.json \
    --excel_output_path ./PeerWiseData/Law/Auckland_law_gpt35_random_100.xlsx \
    --model_name gpt-3.5-turbo \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_script.py \
    --data_path ./PeerWiseData/Medicine/Medicine_year1_vicuna_13b_random_100.json \
    --output_path ./PeerWiseData/Medicine/Medicine_year1_gpt35_random_100.json \
    --excel_output_path ./PeerWiseData/Medicine/Medicine_year1_gpt35_random_100.xlsx \
    --model_name gpt-3.5-turbo \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_script.py \
    --data_path ./PeerWiseData/Medicine/Medicine_year2_vicuna_13b_random_100.json \
    --output_path ./PeerWiseData/Medicine/Medicine_year2_gpt35_random_100.json \
    --excel_output_path ./PeerWiseData/Medicine/Medicine_year2_gpt35_random_100.xlsx \
    --model_name gpt-3.5-turbo \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_script.py \
    --data_path ./Paul_new_data/Sydney/Sydney_gpt-4_random_100.json \
    --output_path ./Paul_new_data/Sydney/Sydney_gpt-35_random_100_correct.json \
    --excel_output_path ./Paul_new_data/Sydney/Sydney_gpt-35_random_100_correct.xlsx \
    --model_name gpt-3.5-turbo \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY



python gpt-4_evaluation_script.py \
    --data_path ./Paul_new_data/Sydney/evaluator_Test_sydney.json \
    --output_path ./Paul_new_data/Sydney/Sydney_gpt-4_evaluator_Test.json \
    --excel_output_path ./Paul_new_data/Sydney/Sydney_gpt-4_evaluator_Test.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_evaluation_script.py \
    --data_path ./Paul_new_data/Cardiff/evaluator_Test_cardiff.json \
    --output_path ./Paul_new_data/Cardiff/Cardiff_gpt-4_evaluator_Test.json \
    --excel_output_path ./Paul_new_data/Cardiff/Cardiff_gpt-4_evaluator_Test.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY

python gpt-4_evaluation_script.py \
    --data_path ./PeerWiseData/Law/evaluator_Test_auckland_law.json \
    --output_path ./PeerWiseData/Law/Law_gpt-4_evaluator_Test.json \
    --excel_output_path ./PeerWiseData/Law/Law_gpt-4_evaluator_Test.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY

python gpt-4_evaluation_script.py \
    --data_path ./PeerWiseData/Medicine/evaluator_Test_uk_medical_year1.json \
    --output_path ./PeerWiseData/Medicine/uk_medical_year1_gpt-4_evaluator_Test.json \
    --excel_output_path ./PeerWiseData/Medicine/uk_medical_year1_gpt-4_evaluator_Test.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY


python gpt-4_evaluation_script.py \
    --data_path ./PeerWiseData/Medicine/evaluator_Test_uk_medical_year2.json \
    --output_path ./PeerWiseData/Medicine/uk_medical_year2_gpt-4_evaluator_Test.json \
    --excel_output_path ./PeerWiseData/Medicine/uk_medical_year2_gpt-4_evaluator_Test.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY



python gpt-4_script.py \
    --data_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/generator_merged_avg_3_lenexp_10_sample_5000.json \
    --output_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/GPT-4_generator_merged_avg_3_lenexp_10_sample_5000.json \
    --excel_output_path ./Paul_new_data/Merged_Sydney_Cardiff_Law_Medical_Y1_Y2/GPT-4_generator_merged_avg_3_lenexp_10_sample_5000.xlsx \
    --model_name gpt-4 \
    --temperature 0.7 \
    --max_tokens 512 \
    --top_p 1 \
    --frequency_penalty 0 \
    --presence_penalty 0 \
    --api_key YOUR_OPENAI_API_KEY