"""PPO 探针：用一个小假环境跑固定轮数，打印可比数字。

基线 = 仓库自带的真实训练配置（tests/test_ppo_runner.py 第 140-166 行 train_cfg）：
    num_learning_epochs=4, num_mini_batches=4, clip_param=0.2,
    gamma=0.99, lam=0.95, value_loss_coef=1.0, entropy_coef=0.01,
    learning_rate=3e-4, max_grad_norm=1.0, use_clipped_value_loss=True
    ActorCritic: 隐层 [64,64], elu, init_noise_std=1.0
    runner: num_envs=4, max_episode_length=200, num_steps_per_env=100

原则：不传的参数一律沿用基线，绝不偷偷覆盖。
用法（VS Code 终端，在 ~/code/rsl_rl_isrc 下）：
    python ppo_probe.py                 # 基线原样
    python ppo_probe.py --clip 0.5      # 只改 clip
    python ppo_probe.py --epochs 16     # 只改 num_learning_epochs
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsl_rl_isrc.env import VecEnv
from rsl_rl_isrc.modules import ActorCritic
from rsl_rl_isrc.algorithms.ppo import PPO

BASE = dict(
    num_learning_epochs=4,
    num_mini_batches=4,
    clip_param=0.2,
    gamma=0.99,
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.01,
    learning_rate=3e-4,
    max_grad_norm=1.0,
    use_clipped_value_loss=True,
)


class TinyEnv(VecEnv):
    """4 维观测、1 维动作的假环境：只为让 PPO 有数可算，不追求物理正确。"""

    def __init__(self, num_envs=4, max_episode_length=200, seed=0):
        self.num_envs = num_envs
        self.num_obs = 4
        self.num_privileged_obs = None
        self.num_actions = 1
        self.max_episode_length = max_episode_length
        self.device = torch.device("cpu")
        self.gen = torch.Generator().manual_seed(seed)
        self.obs_buf = torch.zeros(num_envs, 4)
        self.rew_buf = torch.zeros(num_envs)
        self.reset_buf = torch.ones(num_envs, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.int32)
        self.reset(torch.arange(num_envs))

    def step(self, actions):
        cart_pos, cart_vel, pole_angle, pole_vel = self.obs_buf[:, 0], self.obs_buf[:, 1], self.obs_buf[:, 2], self.obs_buf[:, 3]
        force = actions[:, 0]
        cart_vel = cart_vel + 0.1 * force
        cart_vel = cart_vel * 0.99
        cart_pos = cart_pos + cart_vel
        acc = -9.8 * torch.sin(pole_angle) - 0.1 * cart_vel * torch.cos(pole_angle) + 0.1 * force * torch.cos(pole_angle)
        pole_vel = (pole_vel + acc) * 0.99
        pole_angle = pole_angle + pole_vel

        self.obs_buf[:, 0] = cart_pos
        self.obs_buf[:, 1] = cart_vel
        self.obs_buf[:, 2] = pole_angle
        self.obs_buf[:, 3] = pole_vel

        rewards = torch.ones(self.num_envs)
        self.episode_length_buf = self.episode_length_buf + 1
        done = (cart_pos.abs() > 2.4) | (pole_angle.abs() > 0.2094) | (self.episode_length_buf >= self.max_episode_length)

        ids = torch.where(done)[0]
        if len(ids) > 0:
            self.reset(ids)
        return self.obs_buf.clone(), None, rewards, done, {}

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs)
        n = self.num_envs * 4
        noise = (torch.rand(n, generator=self.gen) - 0.5) * 0.1
        noise = noise.reshape(self.num_envs, 4)
        self.obs_buf[env_ids] = noise[env_ids]
        self.episode_length_buf[env_ids] = 0
        return self.obs_buf.clone()

    def get_observations(self):
        return self.obs_buf.clone()

    def get_privileged_observations(self):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--mini", type=int, default=None)
    ap.add_argument("--entropy", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--lam", type=float, default=None)
    ap.add_argument("--grad-norm", type=float, default=None)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="baseline")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    env = TinyEnv(num_envs=4, max_episode_length=200, seed=args.seed)
    ac = ActorCritic(env.num_obs, env.num_obs, env.num_actions,
                     actor_hidden_dims=[64, 64], critic_hidden_dims=[64, 64],
                     activation="elu", init_noise_std=1.0)

    alg_kwargs = dict(BASE)
    if args.clip is not None:
        alg_kwargs["clip_param"] = args.clip
    if args.lr is not None:
        alg_kwargs["learning_rate"] = args.lr
    if args.epochs is not None:
        alg_kwargs["num_learning_epochs"] = args.epochs
    if args.mini is not None:
        alg_kwargs["num_mini_batches"] = args.mini
    if args.entropy is not None:
        alg_kwargs["entropy_coef"] = args.entropy
    if args.gamma is not None:
        alg_kwargs["gamma"] = args.gamma
    if args.lam is not None:
        alg_kwargs["lam"] = args.lam
    if args.grad_norm is not None:
        alg_kwargs["max_grad_norm"] = args.grad_norm

    alg = PPO(ac, device="cpu", **alg_kwargs)
    alg.init_storage(env.num_envs, args.steps, [env.num_obs], [None], [env.num_actions])

    print("TAG={} clip={} lr={} epochs={} mini={} entropy={} gamma={} lam={} grad_norm={} iters={} steps={} seed={}".format(
        args.tag, alg.clip_param, alg.learning_rate, alg.num_learning_epochs, alg.num_mini_batches,
        alg.entropy_coef, alg.gamma, alg.lam, alg.max_grad_norm, args.iters, args.steps, args.seed))

    obs = env.get_observations()
    critic_obs = obs
    ep_ret = torch.zeros(env.num_envs)
    finished = []

    for it in range(1, args.iters + 1):
        for _ in range(args.steps):
            actions = alg.act(obs, critic_obs)
            obs, _, rewards, dones, infos = env.step(actions)
            critic_obs = obs
            alg.process_env_step(rewards, dones, infos)
            ep_ret = ep_ret + rewards
            ids = torch.where(dones)[0]
            if len(ids) > 0:
                finished.extend(ep_ret[ids].tolist())
                ep_ret[ids] = 0.0
        alg.compute_returns(critic_obs)
        value_loss, surrogate_loss = alg.update()

        if it % 50 == 0 or it == args.iters:
            recent = finished[-20:]
            ret_txt = "{:.2f}".format(sum(recent) / len(recent)) if recent else "n/a"
            print("  it={:4d} surrogate={:+.4f} value={:.2f} 回报={} std={:.3f}".format(
                it, surrogate_loss, value_loss, ret_txt, ac.std.mean().item()))


if __name__ == "__main__":
    main()
