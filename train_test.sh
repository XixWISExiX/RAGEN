set -e

# Algorithm Settings
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.

# Hyperparameter Settings
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"

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

#---------------------- OUNLP -----------------------

# Batch Sizes Experiments
debug_name="ppo-basic-medium-batch_size-1/32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=1 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

debug_name="ppo-basic-medium-batch_size-2/32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=2 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

debug_name="ppo-basic-medium-batch_size-4/32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=4 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

wait

debug_name="ppo-basic-medium-batch_size-8/32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=8 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

debug_name="ppo-basic-medium-batch_size-16/32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=16 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

debug_name="ppo-basic-medium-batch_size-32/32"
python train.py --config-name _3_frozen_lake trainer.project_name=frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_name $USE_PPO $USE_BASE $USE_MEDIUM micro_batch_size_per_gpu=32 ppo_mini_batch_size=32 > logs/$debug_name.out 2> logs/$debug_name.err &

#----------------------------------------------------











































# NO MANSLAND!


# Section 3.1&3.2 - General Observations
#python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=bandit-ppo $USE_PPO $USE_BASE $USE_SMALL > logs/bandit-ppo.out 2> logs/bandit-ppo.err &
#python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=bandit-grpo $USE_GRPO $USE_BASE $USE_SMALL > logs/bandit-grpo.out 2> logs/bandit-grpo.err &
#python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-ppo $USE_PPO $USE_BASE $USE_SMALL > logs/sokoban-ppo.out 2> logs/sokoban-ppo.err &
#python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-grpo $USE_GRPO $USE_BASE $USE_SMALL > logs/sokoban-grpo.out 2> logs/sokoban-grpo.err &
# Can only train 4 at a time

## FROZEN LAKE
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=frozen_lake-ppo $USE_PPO $USE_BASE $USE_SMALL > logs/frozen_lake-ppo.out 2> logs/frozen_lake-ppo.err &
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=frozen_lake-grpo $USE_GRPO $USE_BASE $USE_SMALL > logs/frozen_lake-grpo.out 2> logs/frozen_lake-grpo.err &

#debug_1_name="debug-output-1"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=$debug_1_name $USE_GRPO $USE_BASE $USE_SMALL > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
#debug_2_name="debug-2GPU-1"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"0,1"' trainer.experiment_name=$debug_2_name $USE_PPO $USE_BASE $USE_SMALL_2_GPU > logs/$debug_2_name.out 2> logs/$debug_2_name.err &
#debug_4_name="debug-4GPU-1"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"0,1,2,3"' trainer.experiment_name=$debug_4_name $USE_PPO $USE_BASE $USE_SMALL_4_GPU > logs/$debug_4_name.out 2> logs/$debug_4_name.err &
#debug_1_name="debug-batch-2"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL micro_batch_size_per_gpu=2 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
#debug_1_name="debug-batch-4"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL micro_batch_size_per_gpu=4 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
#NOTE
#debug_1_name="debug-batch-8"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"0"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL micro_batch_size_per_gpu=8 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
# TODO, test below metrics (big & small)
#NOTE This doesn't work because the rollout worker never gets a chance to build a multi-turn trajectory
#debug_1_name="debug-model_len-1012"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL actor_rollout_ref.rollout.max_model_len=1012 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
#debug_1_name="debug-model_len-3060"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL actor_rollout_ref.rollout.max_model_len=3060 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
#debug_1_name="debug-model_len-4096"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"3"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL actor_rollout_ref.rollout.max_model_len=4096 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
# actor_rollout_ref.rollout.max_model_len=2048

#debug_1_name="debug-response_len-64"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"0"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL actor_rollout_ref.rollout.response_length=64 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
#debug_1_name="debug-response_len-256"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL actor_rollout_ref.rollout.response_length=256 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &
# actor_rollout_ref.rollout.response_length=128

#debug_1_name="debug-model_len-4096-response_len-256"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"0"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL actor_rollout_ref.rollout.max_model_len=4096 actor_rollout_ref.rollout.response_length=256 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &

#debug_1_name="debug-gpu-mem-0.9"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL actor_rollout_ref.rollout.gpu_memory_utilization=0.9> logs/$debug_1_name.out 2> logs/$debug_1_name.err &

#debug_1_name="debug-medium"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"2"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_MEDIUM > logs/$debug_1_name.out 2> logs/$debug_1_name.err &

#debug_1_name="debug-medium-batch-4-2GPU"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"1,3"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_2_GPU micro_batch_size_per_gpu=4 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &

# TODO 
#debug_1_name="debug-large-2GPU"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"2,3"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_LARGE $USE_2_GPU micro_batch_size_per_gpu=4 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &

#debug_1_name="debug-qwen-7B-small-4GPU"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"0,1,2,3"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_4_GPU_MODEL $USE_QWEN_7B micro_batch_size_per_gpu=1 ppo_mini_batch_size=4> logs/$debug_1_name.out 2> logs/$debug_1_name.err &

# NOTE
# n_gpu * micro_batch_size_per_gpu = batch_size
# batch_size / ppo_mini_batch_size = n_gradient_updates

#debug_1_name="debug-qwen-3B-small-2GPU-Model"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"2,3"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL $USE_2_GPU_MODEL $USE_QWEN_3B micro_batch_size_per_gpu=1 ppo_mini_batch_size=4 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &

#debug_1_name="debug-qwen-7B-small-2GPU-Model-2GPU-other"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"0,1,2,3"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL $USE_2_GPU_MODEL trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tp_size_check=False $USE_QWEN_7B micro_batch_size_per_gpu=1 ppo_mini_batch_size=4> logs/$debug_1_name.out 2> logs/$debug_1_name.err &

#debug_1_name="debug-llama2-4GPU"
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"0,1,2,3"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_MEDIUM $USE_4_GPU micro_batch_size_per_gpu=4 $USE_LLAMA2_8B > logs/$debug_1_name.out 2> logs/$debug_1_name.err &

#wait


# Section 3.1&3.2 - General Observations
#python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=bandit-ppo $USE_PPO $USE_BASE &
#python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=bandit-grpo $USE_GRPO $USE_BASE &
#python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-ppo $USE_PPO $USE_BASE &
#python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-grpo $USE_GRPO $USE_BASE &
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="4" trainer.experiment_name=frozen_lake-ppo $USE_PPO $USE_BASE &
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=frozen_lake-grpo $USE_GRPO $USE_BASE &


