#!/bin/sh
env="UCMEC"  # MPE에서 UCMEC으로 변경
scenario="MA_UCMEC_stat_ldmu"  # 환경 클래스명으로 변경
num_landmarks=3  # UCMEC 환경에서는 불필요하지만 유지
num_agents=10  # UCMEC 환경의 M_sim 값과 맞춰야 함
algo="rmappo"
exp="ucmec_stat_ldmu"  # 실험명 변경
seed_max=10

# 전력 페널티 함수 설정
power_penalty="((np.exp(self.p_last[i])-1)/100)"

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
echo "power penalty function: ${power_penalty}"

for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    PYTHONPATH=.. CUDA_VISIBLE_DEVICES=0 python ../train/train_wandb.py --use_valuenorm \
    --use_popart \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents ${num_agents} \
    --num_landmarks ${num_landmarks} \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 8 \
    --num_mini_batch 32 \
    --episode_length 256 \
    --num_env_steps 100000 \
    --ppo_epoch 10 \
    --use_ReLU \
    --gain 0.01 \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --user_name "2yunsu" \
    --use_naive_recurrent_policy \
    --use_ippo \
    --use_wandb \
    --power_penalty_func "${power_penalty}"
done