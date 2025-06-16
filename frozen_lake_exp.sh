set -e

# Algorithm Settings
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.

# Hyperparameter Settings
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"
USE_HIGH_KL="algorithm.kl_ctrl.kl_coef=0.01 actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"
USE_LOW_KL="algorithm.kl_ctrl.kl_coef=0.0001 actor_rollout_ref.actor.kl_loss_coef=0.0001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"
USE_HIGH_CLIP_SYMMETRIC="algorithm.kl_ctrl.kl_coef=0.01 actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.clip_ratio_high=0.4 actor_rollout_ref.actor.clip_ratio_low=0.4 actor_rollout_ref.rollout.rollout_filter_ratio=1"
USE_LOW_CLIP_SYMMETRIC="algorithm.kl_ctrl.kl_coef=0.01 actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.clip_ratio_high=0.1 actor_rollout_ref.actor.clip_ratio_low=0.1 actor_rollout_ref.rollout.rollout_filter_ratio=1"
USE_MEDUIM_FILTER="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=0.5"
USE_NO_FILTER="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=0"
USE_HIGH_CLIP_ASYMMETRIC="algorithm.kl_ctrl.kl_coef=0.01 actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.clip_ratio_high=0.4 actor_rollout_ref.actor.clip_ratio_low=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"
USE_LOW_CLIP_ASYMMETRIC="algorithm.kl_ctrl.kl_coef=0.01 actor_rollout_ref.actor.kl_loss_coef=0.01 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.actor.clip_ratio_low=0.4 actor_rollout_ref.rollout.rollout_filter_ratio=1"


# Model Lengths
USE_MINI="actor_rollout_ref.rollout.max_model_len=1024 actor_rollout_ref.rollout.response_length=128"
USE_SMALL="actor_rollout_ref.rollout.max_model_len=2048 actor_rollout_ref.rollout.response_length=128"
USE_MEDIUM="actor_rollout_ref.rollout.max_model_len=4096 actor_rollout_ref.rollout.response_length=256"
USE_LARGE="actor_rollout_ref.rollout.max_model_len=6144 actor_rollout_ref.rollout.response_length=384"

# PPO mini batch sizes
USE_PPO_MINI_BATCH="micro_batch_size_per_gpu=1 ppo_mini_batch_size=4"
USE_PPO_SMALL_BATCH="micro_batch_size_per_gpu=2 ppo_mini_batch_size=8"
USE_PPO_MEDIUM_BATCH="micro_batch_size_per_gpu=4 ppo_mini_batch_size=16"
USE_PPO_LARGE_BATCH="micro_batch_size_per_gpu=8 ppo_mini_batch_size=32"

# Number of GPUs to load model
USE_2_GPU_MODEL="trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2"
USE_4_GPU_MODEL="trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4"

# NOTE not tested
# Model Type/Size Settings
USE_LLAMA2_8B="model_path=meta-llama/Meta-Llama-3-8B-Instruct enable_response_mask=False"
USE_LLAMA2_70B="model_path=meta-llama/Llama-2-70b enable_response_mask=False"
USE_QWEN_3B="model_path=Qwen/Qwen2.5-3B-Instruct actor_rollout_ref.rollout.gpu_memory_utilization=0.75"
USE_QWEN_7B="model_path=Qwen/Qwen2.5-7B-Instruct actor_rollout_ref.rollout.gpu_memory_utilization=0.75"
USE_QWEN_72B="model_path=Qwen/Qwen2.5-72B-Instruct"

# LoRA (Efficient Training with Low-Rank Adaptation)
LoRA_32_32_ALL="lora.rank=32 lora.alpha=32 lora.target_modules=all-linear"
#LoRA_32_32_ATTENTION="lora.rank=32 lora.alpha=32 lora.target_modules=[q_proj,k_proj,v_proj,o_proj]"

#---------------------- OUNLP -----------------------

# Batch Sizes Experiments (medium)
#debug_name="ppo-basic-medium-batch_size-1-32"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=1 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size-2-32"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=2 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size-4-32"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=4 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &
#
#wait
#
#debug_name="ppo-basic-medium-batch_size-8-32"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=8 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size-16-32"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=16 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size-32-32"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=32 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &


# NEW TEST =====================================================

debug_name="ppo-basic-medium-LoRA-32-32-all-batch_size-1-32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $LoRA_32_32_ALL micro_batch_size_per_gpu=1 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

debug_name="ppo-basic-medium-LoRA-32-32-all-batch_size-2-32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $LoRA_32_32_ALL micro_batch_size_per_gpu=2 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

debug_name="ppo-basic-medium-LoRA-32-32-all-batch_size-4-32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $LoRA_32_32_ALL micro_batch_size_per_gpu=4 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

wait

debug_name="ppo-basic-medium-LoRA-32-32-all-batch_size-8-32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $LoRA_32_32_ALL micro_batch_size_per_gpu=8 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

debug_name="ppo-basic-medium-LoRA-32-32-all-batch_size-16-32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $LoRA_32_32_ALL micro_batch_size_per_gpu=16 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

debug_name="ppo-basic-medium-LoRA-32-32-all-batch_size-32-32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $LoRA_32_32_ALL micro_batch_size_per_gpu=32 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

wait

#debug_name="ppo-basic-medium-batch_size_mini"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_PPO_MINI_BATCH > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size_mini-high_kl"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_PPO_MINI_BATCH $USE_HIGH_KL > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size_mini-low_kl"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_PPO_MINI_BATCH $USE_LOW_KL > logs/$debug_name.out 2> logs/$debug_name.err &
#
#wait
#
#debug_name="ppo-basic-medium-batch_size_mini-high_clip_symmetric"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_PPO_MINI_BATCH $USE_HIGH_CLIP_SYMMETRIC > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size_mini-low_clip_symmetric"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_PPO_MINI_BATCH $USE_LOW_CLIP_SYMMETRIC > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size_mini-medium_filter"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_PPO_MINI_BATCH $USE_MEDIUM_FILTER > logs/$debug_name.out 2> logs/$debug_name.err &
#
#wait
#
#debug_name="ppo-basic-medium-batch_size_mini-no_filter"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_PPO_MINI_BATCH $USE_NO_FILTER > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size_mini-high_clip_asymmetric"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_PPO_MINI_BATCH $USE_HIGH_CLIP_ASYMMETRIC > logs/$debug_name.out 2> logs/$debug_name.err &
#
#debug_name="ppo-basic-medium-batch_size_mini-low_clip_asymmetric"
#python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_PPO_MINI_BATCH $USE_LOW_CLIP_ASYMMETRIC > logs/$debug_name.out 2> logs/$debug_name.err &

#----------------------------------------------------
