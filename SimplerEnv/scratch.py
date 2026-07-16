for i in range(40):
    test_video_path = f"{i}_{self.instruction[i]}.mp4"
    imageio.mimwrite(test_video_path, self.obs[:,i], fps=10) 



for name, param in decoder_layer.named_parameters():
    if param.requires_grad and param.dim() > 1:  # usually skip bias (1D)
        print(f"{name}: sum={param.data.sum().item():.6f}")

for name, param in self.policy.vla.named_parameters():
    if param.grad is not None:   # 只看有梯度的
        print(f"{name}.grad:\n", param.grad.sum())

for name, param in self.policy.vla.named_parameters():
    if param.requires_grad and param.dim() > 1:  # usually skip bias (1D)
        print(f"{name}: sum={param.data.sum().item():.6f}")

for name, param in self.alg.policy.vla.named_parameters():
    if param.requires_grad and param.dim() > 1:  # usually skip bias (1D)
        print(f"{name}: sum={param.data.sum().item():.6f}")


base_model.model.language_model.model.layers.30.mlp.down_proj.lora_B.default.weight: sum=-0.047468
base_model.model.language_model.model.layers.31.self_attn.q_proj.lora_A.default.weight: sum=20.668446
base_model.model.language_model.model.layers.31.self_attn.q_proj.lora_B.default.weight: sum=-0.005704
base_model.model.language_model.model.layers.31.self_attn.k_proj.lora_A.default.weight: sum=31.064323
base_model.model.language_model.model.layers.31.self_attn.k_proj.lora_B.default.weight: sum=-0.004064
base_model.model.language_model.model.layers.31.self_attn.v_proj.lora_A.default.weight: sum=5.059812
base_model.model.language_model.model.layers.31.self_attn.v_proj.lora_B.default.weight: sum=-0.030488
base_model.model.language_model.model.layers.31.self_attn.o_proj.lora_A.default.weight: sum=16.653442
base_model.model.language_model.model.layers.31.self_attn.o_proj.lora_B.default.weight: sum=-0.040641
base_model.model.language_model.model.layers.31.mlp.gate_proj.lora_A.default.weight: sum=19.006052
base_model.model.language_model.model.layers.31.mlp.gate_proj.lora_B.default.weight: sum=0.064349
base_model.model.language_model.model.layers.31.mlp.up_proj.lora_A.default.weight: sum=2.726670
base_model.model.language_model.model.layers.31.mlp.up_proj.lora_B.default.weight: sum=0.002225
base_model.model.language_model.model.layers.31.mlp.down_proj.lora_A.default.weight: sum=13.019191
base_model.model.language_model.model.layers.31.mlp.down_proj.lora_B.default.weight: sum=-0.013370
base_model.model.language_model.lm_head.lora_A.default.weight: sum=2.977984
base_model.model.language_model.lm_head.lora_B.default.weight: sum=-0.005152
base_model.model.value_head.head_l1.weight: sum=25.250000
base_model.model.value_head.head_l2.weight: sum=3.078125
base_model.model.value_head.head_l3.weight: sum=-0.192383

optimizer 之后： 
base_model.model.language_model.model.layers.30.mlp.up_proj.lora_A.default.weight: sum=-12.320019
base_model.model.language_model.model.layers.30.mlp.up_proj.lora_B.default.weight: sum=0.127572
base_model.model.language_model.model.layers.30.mlp.down_proj.lora_A.default.weight: sum=15.777561
base_model.model.language_model.model.layers.30.mlp.down_proj.lora_B.default.weight: sum=-0.054292
base_model.model.language_model.model.layers.31.self_attn.q_proj.lora_A.default.weight: sum=20.680698
base_model.model.language_model.model.layers.31.self_attn.q_proj.lora_B.default.weight: sum=-0.007448
base_model.model.language_model.model.layers.31.self_attn.k_proj.lora_A.default.weight: sum=31.056456
base_model.model.language_model.model.layers.31.self_attn.k_proj.lora_B.default.weight: sum=-0.026050
base_model.model.language_model.model.layers.31.self_attn.v_proj.lora_A.default.weight: sum=5.060857
base_model.model.language_model.model.layers.31.self_attn.v_proj.lora_B.default.weight: sum=-0.041133
base_model.model.language_model.model.layers.31.self_attn.o_proj.lora_A.default.weight: sum=16.626852
base_model.model.language_model.model.layers.31.self_attn.o_proj.lora_B.default.weight: sum=-0.015715
base_model.model.language_model.model.layers.31.mlp.gate_proj.lora_A.default.weight: sum=18.953442
base_model.model.language_model.model.layers.31.mlp.gate_proj.lora_B.default.weight: sum=0.108771
base_model.model.language_model.model.layers.31.mlp.up_proj.lora_A.default.weight: sum=2.709093
base_model.model.language_model.model.layers.31.mlp.up_proj.lora_B.default.weight: sum=-0.027521
base_model.model.language_model.model.layers.31.mlp.down_proj.lora_A.default.weight: sum=13.059200
base_model.model.language_model.model.layers.31.mlp.down_proj.lora_B.default.weight: sum=-0.019739
base_model.model.language_model.lm_head.lora_A.default.weight: sum=2.952590
base_model.model.language_model.lm_head.lora_B.default.weight: sum=-0.011216
base_model.model.value_head.head_l1.weight: sum=25.250000
base_model.model.value_head.head_l2.weight: sum=3.078125
base_model.model.value_head.head_l3.weight: sum=-0.192383


value head的没变， 说明optimizer只更新了vla的lora参数。 很棒！



4次初始化：
initialization: 109715987
initialization: 160079515
initialization: 107251574
initialization: 111705138

initialization: 109715987
initialization: 160079515
initialization: 107251574
initialization: 111705138
initialization: 134918602
initialization: 126779787
initialization: 134838547
initialization: 127427578
initialization: 133905680
initialization: 113058067


initialization: 109715987
initialization: 160079515
initialization: 107251574
initialization: 111663784    这个不知道为什么不一样， 但是其他的都一样就行了
initialization: 134918602
initialization: 126779787
initialization: 134838547
initialization: 127427578
initialization: 133905680
initialization: 113058067




/home/cldb5/.conda/envs/rlvla_env3/bin/python  -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path openvla/openvla-7b --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 320 --alg_name ppo --ttt 0 --max_episodes 40 

/home/cldb5/.conda/envs/rlvla_env3/bin/python  /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path openvla/openvla-7b --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 320 --alg_name ppo --ttt 0 --max_episodes 40 --normalize_advantage false 



no warmup
/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path openvla/openvla-7b --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 320 --alg_name ppo --ttt 1 --max_episodes 20 --normalize_advantage --buffer_gamma 0.99 


/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path openvla/openvla-7b --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 320 --alg_name ppo --ttt 1 --max_episodes 20 --normalize_advantage --buffer_gamma 0


/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path openvla/openvla-7b --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 320 --alg_name ppo --ttt 1 --max_episodes 20 --no-normalize-advantage --buffer_gamma 0


/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path openvla/openvla-7b --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 320 --alg_name ppo --ttt 0 --max_episodes 40 


warmup
/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path /storage/1tb/changyu/pretrain_models/openvla-7b-rlvla-warmup --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 160 --alg_name ppo --ttt 1 --max_episodes 20 --normalize_advantage --buffer_gamma 0.99 


/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path /storage/1tb/changyu/pretrain_models/openvla-7b-rlvla-warmup --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 160 --alg_name ppo --ttt 1 --max_episodes 20 --normalize_advantage --buffer_gamma 0


/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path /storage/1tb/changyu/pretrain_models/openvla-7b-rlvla-warmup --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 160 --alg_name ppo --ttt 1 --max_episodes 80 --no-normalize-advantage --buffer_gamma 0 --from_epoch 40


/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path /storage/1tb/changyu/pretrain_models/openvla-7b-rlvla-warmup --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 160 --alg_name ppo --ttt 0 --max_episodes 80 --from_epoch 40


no warmup
/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path openvla/openvla-7b  --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 160 --alg_name ppo --ttt 1 --max_episodes 40 --no-normalize-advantage --buffer_gamma 0


/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path openvla/openvla-7b  --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 160 --alg_name ppo --ttt 0 --max_episodes 40 



warmup: tt_step = 16  
/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path /storage/1tb/changyu/pretrain_models/openvla-7b-rlvla-warmup --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 168 --alg_name ppo --ttt 1 --max_episodes 40 --normalize_advantage --buffer_gamma 0.99 --tt_steps 12



/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path /storage/1tb/changyu/pretrain_models/openvla-7b-rlvla-warmup --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 160 --alg_name ppo --ttt 1 --max_episodes 40 --no-normalize-advantage --buffer_gamma 0 --tt_steps 4

/home/cldb5/.conda/envs/rlvla_env3/bin/python -- /home/cldb5/RL4VLA/SimplerEnv/simpler_env/train_ms3_ppo_ttt.py --name PPO-pc25m_v3-warmup --env_id PutOnPlateInScene25MultiCarrot-v1 --vla_path /storage/1tb/changyu/pretrain_models/openvla-7b-rlvla-warmup --vla_unnorm_key bridge_orig --seed 0 --num_envs 1 --episode_len 160 --alg_name ppo --ttt 1 --max_episodes 40 --no-normalize-advantage --buffer_gamma 0 --tt_steps 2


