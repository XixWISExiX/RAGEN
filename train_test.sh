set -e

# Section 1: Base Experiments
USE_GRPO="algorithm.adv_estimator=grpo agent_proxy.reward_normalization.method=mean_std actor_rollout_ref.actor.use_kl_loss=True"
USE_PPO="algorithm.adv_estimator=gae" # by default.
USE_BASE="algorithm.kl_ctrl.kl_coef=0.001 actor_rollout_ref.actor.kl_loss_coef=0.001 actor_rollout_ref.actor.clip_ratio_high=0.2 actor_rollout_ref.rollout.rollout_filter_ratio=1"
USE_SMALL="micro_batch_size_per_gpu=1 ppo_mini_batch_size=8 actor_rollout_ref.rollout.max_model_len=2048 actor_rollout_ref.rollout.response_length=128"
USE_SMALL_2_GPU="trainer.n_gpus_per_node=2 actor_rollout_ref.rollout.tensor_model_parallel_size=2 micro_batch_size_per_gpu=1 ppo_mini_batch_size=8 actor_rollout_ref.rollout.max_model_len=2048 actor_rollout_ref.rollout.response_length=128"
# trainer.nnodes is the number of compute nodes
# actor_rollout_ref.rollout.n is the number of actor models
# TODO still needs to debug
USE_SMALL_4_GPU="trainer.n_gpus_per_node=4 actor_rollout_ref.rollout.tensor_model_parallel_size=4 micro_batch_size_per_gpu=1 ppo_mini_batch_size=8 actor_rollout_ref.rollout.max_model_len=2048 actor_rollout_ref.rollout.response_length=128"
USE_MEDIUM="micro_batch_size_per_gpu=1 ppo_mini_batch_size=16 actor_rollout_ref.rollout.max_model_len=4096 actor_rollout_ref.rollout.response_length=256"

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

debug_1_name="debug-model_len-4096-response_len-256"
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"0"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_SMALL actor_rollout_ref.rollout.max_model_len=4096 actor_rollout_ref.rollout.response_length=256 > logs/$debug_1_name.out 2> logs/$debug_1_name.err &

debug_1_name="debug-medium"
python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES='"1"' trainer.experiment_name=$debug_1_name $USE_PPO $USE_BASE $USE_MEDIUM > logs/$debug_1_name.out 2> logs/$debug_1_name.err &



#wait


# Section 3.1&3.2 - General Observations
#python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="0" trainer.experiment_name=bandit-ppo $USE_PPO $USE_BASE &
#python train.py --config-name _1_bandit system.CUDA_VISIBLE_DEVICES="1" trainer.experiment_name=bandit-grpo $USE_GRPO $USE_BASE &
#python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="2" trainer.experiment_name=sokoban-ppo $USE_PPO $USE_BASE &
#python train.py --config-name _2_sokoban system.CUDA_VISIBLE_DEVICES="3" trainer.experiment_name=sokoban-grpo $USE_GRPO $USE_BASE &
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="4" trainer.experiment_name=frozen_lake-ppo $USE_PPO $USE_BASE &
#python train.py --config-name _3_frozen_lake system.CUDA_VISIBLE_DEVICES="5" trainer.experiment_name=frozen_lake-grpo $USE_GRPO $USE_BASE &


