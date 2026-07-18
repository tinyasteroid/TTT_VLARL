import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import imageio
import os

from simpler_env.utils.synthetic_rewards import (
    EPISODES_PER_TASK,
    STEPS_PER_EPISODE,
    build_task_reward_schedule,
)


class SeparatedReplayBuffer(object):
    def __init__(self, all_args, obs_dim, act_dim):
        self.ep_len = all_args.episode_len
        self.num_env = all_args.num_envs
        self.gamma = all_args.buffer_gamma
        self.gae_lambda = all_args.buffer_lambda
        self.buffer_minibatch = all_args.buffer_minibatch
        self.alg_grpo_fix = all_args.alg_grpo_fix

        self.obs = np.zeros((self.ep_len + 1, self.num_env, *obs_dim), dtype=np.uint8)
        self.instruction = [""] * self.num_env
        self.value_preds = np.zeros((self.ep_len + 1, self.num_env, 1), dtype=np.float32)
        self.returns = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.actions = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.int32)
        self.action_log_probs = np.zeros((self.ep_len, self.num_env, act_dim), dtype=np.float32)
        self.rewards = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.masks = np.ones((self.ep_len + 1, self.num_env, 1), dtype=np.float32)

        self.advantages = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)

        self.step = 0
        self.act_dim = act_dim

    def insert(self, obs, actions, action_log_probs, value_preds, rewards, masks):
        self.obs[self.step + 1] = obs.copy()
        #for some models with different output tokens. 
        action_token_len = actions.shape[-1]
        if action_token_len > self.act_dim:
            action_token_len = self.act_dim
        self.actions[self.step, :, :action_token_len] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        #self.step = (self.step + 1) % self.ep_len
        self.step = (self.step + 1)

    def warmup(self, obs, instruction):
        self.obs[0] = obs
        self.instruction = instruction
        self.masks[0] = 1.0

        self.step = 0

    def endup(self, next_value):
        self.value_preds[-1] = next_value

    def compute_returns_ppo(self):
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            vt1 = self.value_preds[step + 1]
            vt = self.value_preds[step]

            delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + vt

        # calc adv
        advantages = self.returns - self.value_preds[:-1]
        mean_advantages = advantages.mean()
        std_advantages = advantages.std()
        self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)


    def compute_returns_grpo(self):
        if self.alg_grpo_fix:
            rewards_valid = self.rewards[self.rewards != 0]
            rewards_norm = self.rewards.copy()
            rewards_norm[rewards_norm != 0] -= rewards_valid.mean()
            rewards_norm[rewards_norm != 0] /= (rewards_valid.std() + 1e-5)
        else:
            rewards_norm = (self.rewards - self.rewards.mean()) / (self.rewards.std() + 1e-5)

        returns = 0
        for step in reversed(range(self.rewards.shape[0])):
            returns = rewards_norm[step] + self.masks[step + 1] * returns
            self.returns[step] = returns

        # calc adv
        self.advantages = self.returns.copy()

    def get_minibatch_count(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        return num_mini_batch

    def feed_forward_generator(self):
        episode_length, n_rollout_threads = self.rewards.shape[:2]
        batch_size = episode_length * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * self.buffer_minibatch:(i + 1) * self.buffer_minibatch] for i in range(num_mini_batch)]

        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        action_logits = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = self.advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_logits_batch = action_logits[indices]
            adv_targ = advantages[indices]

            # instruct
            instruct_indices = indices % n_rollout_threads
            instruct_batch = [self.instruction[i] for i in instruct_indices]

            yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                   old_action_logits_batch, adv_targ)



class SeparatedReplayBuffer_vlm(SeparatedReplayBuffer):
    def __init__(self, all_args, obs_dim, act_dim):
        super().__init__(all_args, obs_dim, act_dim)
        self.tt_steps = all_args.tt_steps
        self.model_path = "OpenGVLab/VideoChat-Flash-Qwen2_5-7B-1M_res224"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).to(torch.bfloat16).to(torch.device("cuda:1"))
        self.image_processor = self.model.get_vision_tower().image_processor
        mm_llm_compress = False # use the global compress or not
        if mm_llm_compress:
            self.model.config.mm_llm_compress = True
            self.model.config.llm_compress_type = "uniform0_attention"
            self.model.config.llm_compress_layer_list = [4, 18]
            self.model.config.llm_image_token_ratio_list = [1, 0.75, 0.25]
        else:
            self.model.config.mm_llm_compress = False
        
        self.max_num_frames = 512
        self.generation_config = dict(
            do_sample=False,
            temperature=0.0,
            max_new_tokens=1024,
            top_p=0.1,
            num_beams=1
        )

    def compute_returns_ppo(self):
        #TODO1: compute the value from vlm according to the obs
        #for step in range(self.rewards.shape[0]):
        for step in range(self.step-1-self.tt_steps, self.step-1):
            #self.value_preds[step] = self.get_value_from_vlm(self.obs[:step+2], self.instruction)  #objs should be step+1， so slice is :step+2
            self.value_preds[step] = step
        #compute the reward by values
        for step in range(self.step-1-self.tt_steps, self.step-1):
            self.rewards[step] = self.value_preds[step+1] - self.value_preds[step]
        #compute the returns and advantages
        gae = 0
        for step in reversed(range(self.step-1-self.tt_steps, self.step-2)):
            vt1 = self.value_preds[step + 1]
            vt = self.value_preds[step]

            delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + vt
        # calc adv
        advantages = self.returns - self.value_preds[:-1]       #only compute the advatange for the updated steps
        advantages = advantages[self.step-1-self.tt_steps:self.step-2]  #only include advantage for the fisrt self.ttsteps-1 in this round
        mean_advantages = advantages.mean()
        std_advantages = advantages.std()
        self.advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

    def get_value_from_vlm(self, obs_seq, instruction):
        #save obs_seq as a mp4 video, obs_seq（narray） with shape [T, 1, H, W, 3]
        obs_seq = obs_seq[:,0]  # [T, H, W, 3]
        video_path = "temp_video.mp4"
        imageio.mimwrite(video_path, obs_seq, fps=8)
        question1 = f"What is the progress of {instruction} in the video? Approaching to the  Please answer in a word from 0% to 100%."
        #question1 = f"How many percent of {text_candidates[i]} is completed? Please answer in one word, such as "
        output1, chat_history = self.model.chat(video_path=video_path, tokenizer=self.tokenizer, user_prompt=question1, return_history=True, max_num_frames=self.max_num_frames, generation_config=self.generation_config)
        #exact the number from output1 with regex
        import re
        match = re.search(r'(\d{1,3})%', output1)
        if match:
            value = float(match.group(1))
        if value > 100:
            value = 100.0
        elif value < 0:
            value = 0.0
        return value


class SeparatedReplayBuffer_vlac(SeparatedReplayBuffer):
    """Legacy VLAC-backed buffer retained for non-ablation entry points."""

    def __init__(self, all_args, obs_dim, act_dim):
        super().__init__(all_args, obs_dim, act_dim)
        self.progress = np.zeros((self.ep_len + 1, self.num_env, 1), dtype=np.float32)
        self.advantages = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.tt_steps = all_args.tt_steps
        self.normalize_advantage = all_args.normalize_advantage
        self.ref_video = None
        self.args = all_args

        from evo_vlac import GAC_model

        self.model_path = all_args.reward_model_path
        self.Critic = GAC_model(tag="critic")
        device_map = "cuda:0" if all_args.ttt == 1 else "cpu"
        self.Critic.init_model(
            model_path=self.model_path,
            model_type="internvl2",
            device_map=device_map,
        )
        self.Critic.temperature = 0.5
        self.Critic.top_k = 1
        self.Critic.set_config()
        self.Critic.set_system_prompt()

    def compute_returns_ppo(self):
        progress = self.get_progress_from_vlac(
            self.obs[:self.step + 1], self.instruction
        )
        for index, value in enumerate(progress):
            self.progress[index] = value
        for step in range(self.step - self.tt_steps, self.step):
            self.rewards[step] = self.progress[step + 1] - self.progress[step]
        self._compute_reward_advantages()

    def _compute_reward_advantages(self):
        gae = 0
        for step in reversed(range(self.step - self.tt_steps, self.step)):
            delta = self.rewards[step]
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae
        advantages = self.returns[self.step - self.tt_steps:self.step]
        if self.normalize_advantage:
            mean_advantages = advantages.mean()
            std_advantages = advantages.std()
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        self.advantages[self.step - self.tt_steps:self.step] = advantages

    def get_progress_from_vlac(self, obs_seq, instruction):
        from evo_vlac.utils.video_tool import compress_video

        obs_seq = obs_seq[:, 0]
        filename = "temp_video_ttt.mp4" if self.args.ttt else "temp_video_no_ttt.mp4"
        test_video_path = os.path.join(self.args.glob_dir, filename)
        imageio.mimwrite(test_video_path, obs_seq, fps=10)
        test_video_compressed = os.path.join(
            os.path.dirname(test_video_path), "test.mp4"
        )
        _, output_fps = compress_video(
            test_video_path,
            test_video_compressed,
            fps=100,
        )
        _, value_list, _, _ = self.Critic.web_trajectory_critic(
            task_description=instruction,
            main_video_path=test_video_compressed,
            reference_video_path=None,
            batch_num=5,
            think=False,
            skip=1,
            rich=False,
            reverse_eval=False,
            output_path="results",
            fps=float(output_fps),
            frame_skip=False,
            done_flag=False,
            in_context_done=False,
            done_threshold=0.9,
            video_output=False,
        )
        return value_list


class SeparatedReplayBuffer_synthetic(SeparatedReplayBuffer):
    def __init__(self, all_args, obs_dim, act_dim):
        super().__init__(all_args, obs_dim, act_dim)
        self.progress = np.zeros((self.ep_len + 1, self.num_env, 1), dtype=np.float32)
        self.advantages = np.zeros((self.ep_len, self.num_env, 1), dtype=np.float32)
        self.tt_steps = all_args.tt_steps
        self.normalize_advantage = all_args.normalize_advantage
        self.ref_video = None
        self.args = all_args
        self.reward_mode = getattr(all_args, "reward_mode", "synthetic_uniform")
        if self.reward_mode not in {"synthetic_matched", "synthetic_uniform"}:
            raise ValueError(f"Unsupported reward_mode: {self.reward_mode}")
        if self.ep_len != STEPS_PER_EPISODE:
            raise ValueError(
                "Synthetic reward experiments require episode_len=160, got "
                f"{self.ep_len}"
            )
        if self.num_env != 1:
            raise ValueError(
                "The fixed 48,000-step reward schedule requires num_envs=1, got "
                f"{self.num_env}"
            )
        if all_args.max_episodes > EPISODES_PER_TASK:
            raise ValueError(
                "Synthetic reward experiments support at most 20 episodes per task, got "
                f"{all_args.max_episodes}"
            )

        reward_seed = getattr(all_args, "reward_seed", all_args.seed)
        reward_task_index = getattr(all_args, "reward_task_index", 0)
        self.synthetic_rewards = build_task_reward_schedule(
            self.reward_mode,
            reward_seed,
            reward_task_index,
        )
        self.synthetic_reward_cursor = 0

    def start_episode(self, episode: int) -> None:
        """Select the deterministic reward segment for an episode or retry."""
        if not 0 <= episode < EPISODES_PER_TASK:
            raise ValueError(f"episode must be in [0, {EPISODES_PER_TASK - 1}]")
        self.synthetic_reward_cursor = episode * self.ep_len

    def get_minibatch_count(self):
        """Return minibatches in one TTT window, not the full episode buffer."""
        batch_size = self.tt_steps * self.num_env
        if self.buffer_minibatch < 0:
            return 1
        if batch_size % self.buffer_minibatch != 0:
            raise ValueError(
                "tt_steps * num_envs must be divisible by buffer_minibatch: "
                f"{self.tt_steps} * {self.num_env} vs {self.buffer_minibatch}"
            )
        return batch_size // self.buffer_minibatch

    def compute_returns_ppo(self):
        if not 0 < self.tt_steps <= self.step <= self.ep_len:
            raise ValueError(
                "Expected 0 < tt_steps <= step <= episode_len, got "
                f"tt_steps={self.tt_steps}, step={self.step}, episode_len={self.ep_len}"
            )

        #let the value be 0， so we don't need to compute the value
        #self.value_preds[:self.step] = 1 - self.progress[:self.step]
        self._assign_synthetic_rewards()

        #compute the returns and advantages
        gae = 0
        for step in reversed(range(self.step-self.tt_steps, self.step)):
            vt1 = self.value_preds[step + 1]    #actually vt1 and vt are 0, and not used
            vt = self.value_preds[step]
            vt1 = 0
            vt = 0

            delta = self.rewards[step] + self.gamma * vt1 * self.masks[step + 1] - vt
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae      #if gamma==0,  gae=reward
            self.returns[step] = gae + vt
        # calc adv
        advantages = self.returns# - self.value_preds[:-1]       #这里是不是导致advantage：前大后小的地方？
        advantages = advantages[self.step-self.tt_steps:self.step]  #only include advantage for the fisrt self.ttsteps-1 in this round
        if self.normalize_advantage:
            mean_advantages = advantages.mean()
            std_advantages = advantages.std()
            self.advantages[self.step-self.tt_steps:self.step] = (advantages - mean_advantages) / (std_advantages + 1e-5)
        else:
            self.advantages[self.step-self.tt_steps:self.step] = advantages

    def _assign_synthetic_rewards(self):
        """Assign the next deterministic window from this task's schedule."""
        window_size = self.tt_steps * self.num_env
        stop = self.synthetic_reward_cursor + window_size
        if stop > self.synthetic_rewards.size:
            raise RuntimeError("Synthetic reward schedule exhausted")
        reward_window = self.synthetic_rewards[self.synthetic_reward_cursor:stop]
        self.rewards[self.step-self.tt_steps:self.step] = reward_window.reshape(
            self.tt_steps,
            self.num_env,
            1,
        )
        self.synthetic_reward_cursor = stop

    def feed_forward_generator(self):     #only generate the data from range(self.step-self.tt_steps, self.step)
        start = self.step - self.tt_steps
        stop = self.step
        n_rollout_threads = self.rewards.shape[1]
        batch_size = self.tt_steps * n_rollout_threads

        if self.buffer_minibatch < 0:
            num_mini_batch = 1
        else:
            assert batch_size % self.buffer_minibatch == 0
            num_mini_batch = batch_size // self.buffer_minibatch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * self.buffer_minibatch:(i + 1) * self.buffer_minibatch] for i in range(num_mini_batch)]

        obs = self.obs[start:stop].reshape(-1, *self.obs.shape[2:])
        actions = self.actions[start:stop].reshape(-1, self.actions.shape[-1])
        value_preds = self.value_preds[start:stop].reshape(-1, 1)
        returns = self.returns[start:stop].reshape(-1, 1)
        masks = self.masks[start:stop].reshape(-1, 1)
        action_logits = self.action_log_probs[start:stop].reshape(
            -1, self.action_log_probs.shape[-1]
        )
        advantages = self.advantages[start:stop].reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            old_action_logits_batch = action_logits[indices]
            adv_targ = advantages[indices]

            # instruct
            instruct_indices = indices % n_rollout_threads
            instruct_batch = [self.instruction[i] for i in instruct_indices]

            yield (obs_batch, instruct_batch, actions_batch, value_preds_batch, return_batch, masks_batch,
                   old_action_logits_batch, adv_targ)


class SeparatedReplayBuffer_perplexity(SeparatedReplayBuffer):
    def __init__(self, all_args, obs_dim, act_dim):
        super().__init__(all_args, obs_dim, act_dim)
        self.tt_steps = all_args.tt_steps
        self.args = all_args
