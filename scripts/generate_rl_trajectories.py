"""
生成 RL 轨迹数据用于训练 M3（具身交互）

Usage:
    python scripts/generate_rl_trajectories.py --env mock --episodes 1000
    python scripts/generate_rl_trajectories.py --env CartPole-v1 --episodes 500
"""

import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from cortex import CORTEXModel, CORTEXEnvWrapper, EnvironmentWrapper


def generate_trajectories(env_id: str, episodes: int, output_file: str, device: str = "cpu"):
    """用随机策略生成轨迹数据。"""
    print(f"[generate_trajectories] 环境: {env_id}, 回合数: {episodes}")

    env = EnvironmentWrapper(env_id)

    # 创建一个小型随机模型作为策略
    model = CORTEXModel(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        max_seq_len=128,
        use_embodied=True,
    )
    model.to(device)

    wrapper = CORTEXEnvWrapper(model, env, device=device)
    trajectories = []

    for ep in range(episodes):
        result = wrapper.run_episode(max_steps=100)
        traj_text = []
        for step in result["trajectory"]:
            traj_text.append(f"[观测] -> [动作{step['action']}] -> [奖励{step['reward']:.2f}]")

        trajectories.append({
            "episode": ep,
            "reward": result["episode_reward"],
            "length": result["episode_length"],
            "text": "\n".join(traj_text),
        })

        if (ep + 1) % 100 == 0:
            avg_reward = sum(t["reward"] for t in trajectories[-100:]) / 100
            print(f"  已完成 {ep+1}/{episodes}, 最近100回合平均奖励: {avg_reward:.2f}")

    # 保存为文本（用于语言模型训练）
    with open(output_file, "w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(f"回合 {t['episode']} | 奖励 {t['reward']:.2f} | 长度 {t['length']}\n")
            f.write(t["text"] + "\n\n")

    print(f"  轨迹已保存: {output_file} ({len(trajectories)} 回合)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="mock", help="环境 ID (mock / CartPole-v1 / ...)")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--output", type=str, default="data/rl_trajectories.txt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_trajectories(args.env, args.episodes, args.output, args.device)


if __name__ == "__main__":
    main()
