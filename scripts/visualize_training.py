"""
AGI-CORTEX 训练过程实时可视化

支持：
- 实时 loss 曲线（基础 + 各模块）
- 自我状态监控（certainty, cognitive_load, emotional_valence）
- 注意力/意识流可视化
- 模块激活热图
- 生成结果对比

Usage:
    # 实时模式（需要配合训练脚本）
    python scripts/visualize_training.py --log_dir outputs/stage0_base --watch

    # 一次性渲染已有训练日志
    python scripts/visualize_training.py --log_dir outputs/stage0_base --render

    # 生成对比报告
    python scripts/visualize_training.py --compare outputs/stage0_base outputs/stage2_reasoning
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from collections import deque

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("[Error] 需要 matplotlib: pip install matplotlib")
    sys.exit(1)

# 可选 tensorboard 日志解析
try:
    from tensorboard.backend.event_processing import event_accumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False


def parse_log_file(log_path):
    """解析训练日志文件，提取指标。"""
    metrics = {
        'steps': [],
        'loss': [],
        'base_loss': [],
        'symbolic_loss': [],
        'self_loss': [],
        'action_loss': [],
        'plasticity_loss': [],
        'certainty': [],
        'cognitive_load': [],
        'lr': [],
        'val_loss': [],
    }

    if not log_path.exists():
        return metrics

    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or 'Step' not in line:
                # 尝试解析验证行
                if 'Validation:' in line:
                    try:
                        parts = line.split('Validation:')[1]
                        loss_part = parts.split('|')[0].strip()
                        val_loss = float(loss_part.split('=')[1].strip())
                        metrics['val_loss'].append(val_loss)
                    except:
                        pass
                continue

            try:
                # Step 1000/10000 | loss=4.12 | base=4.10 | [sym=0.45 | self=0.02] | cert=0.65 | load=0.42 | lr=1.00e-04
                step_part = line.split('Step')[1].split('|')[0].strip()
                step = int(step_part.split('/')[0].strip())
                metrics['steps'].append(step)

                # 提取各项数值
                def extract(key):
                    if key + '=' in line:
                        try:
                            val = line.split(key + '=')[1].split('|')[0].split(']')[0].strip()
                            return float(val)
                        except:
                            return None
                    return None

                for key in ['loss', 'base']:
                    val = extract(key)
                    if val is not None:
                        if key == 'base':
                            metrics['base_loss'].append(val)
                        else:
                            metrics[key].append(val)

                # 模块损失
                if '[' in line and ']' in line:
                    module_part = line.split('[')[1].split(']')[0]
                    for key, target in [('sym', 'symbolic_loss'), ('symbolic', 'symbolic_loss'),
                                        ('self', 'self_loss'), ('act', 'action_loss'),
                                        ('plast', 'plasticity_loss')]:
                        if key + '=' in module_part:
                            try:
                                val = float(module_part.split(key + '=')[1].split('|')[0].strip())
                                metrics[target].append(val)
                            except:
                                pass

                # 自我状态
                cert = extract('cert')
                if cert is not None:
                    metrics['certainty'].append(cert)
                load = extract('load')
                if load is not None:
                    metrics['cognitive_load'].append(load)

                lr = extract('lr')
                if lr is not None:
                    metrics['lr'].append(lr)

            except Exception as e:
                continue

    return metrics


def render_training_dashboard(metrics, output_path, title="Training Dashboard"):
    """生成训练仪表盘图表。"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    steps = metrics.get('steps', [])
    if not steps:
        fig.text(0.5, 0.5, "No data available yet", ha='center', va='center', fontsize=20)
        plt.savefig(output_path, dpi=120, bbox_inches='tight')
        plt.close()
        return

    # 1. Loss 曲线
    ax1 = fig.add_subplot(gs[0, :2])
    if metrics['loss']:
        ax1.plot(steps[:len(metrics['loss'])], metrics['loss'], 'b-', linewidth=1.5, label='Total Loss', alpha=0.8)
    if metrics['base_loss']:
        ax1.plot(steps[:len(metrics['base_loss'])], metrics['base_loss'], 'g--', linewidth=1, label='Base Loss', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 模块损失分解
    ax2 = fig.add_subplot(gs[0, 2])
    module_data = []
    module_labels = []
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for i, (key, label) in enumerate([
        ('symbolic_loss', 'Symbolic'),
        ('self_loss', 'Self-Model'),
        ('action_loss', 'Action'),
        ('plasticity_loss', 'Plasticity')
    ]):
        if metrics[key]:
            ax2.plot(steps[:len(metrics[key])], metrics[key],
                    color=colors[i], linewidth=1.5, label=label, alpha=0.8)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss')
    ax2.set_title('Module Losses')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. 自我状态监控
    ax3 = fig.add_subplot(gs[1, 0])
    if metrics['certainty']:
        ax3.plot(steps[:len(metrics['certainty'])], metrics['certainty'],
                'c-', linewidth=1.5, label='Certainty', alpha=0.8)
        ax3.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
        ax3.fill_between(steps[:len(metrics['certainty'])], 0, metrics['certainty'],
                        alpha=0.1, color='cyan')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Value')
    ax3.set_title('M2: Self-State Certainty')
    ax3.set_ylim(0, 1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 认知负载
    ax4 = fig.add_subplot(gs[1, 1])
    if metrics['cognitive_load']:
        ax4.plot(steps[:len(metrics['cognitive_load'])], metrics['cognitive_load'],
                'm-', linewidth=1.5, label='Cognitive Load', alpha=0.8)
        ax4.axhline(0.8, color='r', linestyle='--', alpha=0.5, label='Overload')
        ax4.fill_between(steps[:len(metrics['cognitive_load'])], 0, metrics['cognitive_load'],
                        alpha=0.1, color='magenta')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Load')
    ax4.set_title('M2: Cognitive Load')
    ax4.set_ylim(0, 1)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 学习率
    ax5 = fig.add_subplot(gs[1, 2])
    if metrics['lr']:
        ax5.plot(steps[:len(metrics['lr'])], metrics['lr'], 'orange', linewidth=1.5)
        ax5.set_xlabel('Step')
        ax5.set_ylabel('LR')
        ax5.set_title('Learning Rate Schedule')
        ax5.set_yscale('log')
        ax5.grid(True, alpha=0.3)

    # 6. 验证损失
    ax6 = fig.add_subplot(gs[2, 0])
    if metrics['val_loss']:
        # 验证点可能不均匀，用 step 索引
        val_steps = np.linspace(steps[0], steps[-1], len(metrics['val_loss']))
        ax6.plot(val_steps, metrics['val_loss'], 'ro-', linewidth=2, markersize=4, label='Val Loss')
        if metrics['loss']:
            ax6.plot(steps[:len(metrics['loss'])], metrics['loss'], 'b-', alpha=0.3, label='Train Loss')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('Loss')
    ax6.set_title('Validation vs Training')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    # 7. 训练统计
    ax7 = fig.add_subplot(gs[2, 1:])
    ax7.axis('off')

    latest_loss = f"{metrics['loss'][-1]:.4f}" if metrics['loss'] else 'N/A'
    best_val = f"{min(metrics['val_loss']):.4f}" if metrics['val_loss'] else 'N/A'
    avg_cert = f"{np.mean(metrics['certainty']):.3f}" if metrics['certainty'] else 'N/A'
    avg_load = f"{np.mean(metrics['cognitive_load']):.3f}" if metrics['cognitive_load'] else 'N/A'

    stats_text = f"""
    Training Statistics
    {'='*40}
    Total Steps: {max(steps) if steps else 0}
    Latest Loss: {latest_loss}
    Best Val Loss: {best_val}
    Avg Certainty: {avg_cert}
    Avg Cognitive Load: {avg_load}

    Module Activity
    {'='*40}
    Symbolic Loss: {'Active' if metrics['symbolic_loss'] else 'Inactive'}
    Self-Model Loss: {'Active' if metrics['self_loss'] else 'Inactive'}
    Action Loss: {'Active' if metrics['action_loss'] else 'Inactive'}
    Plasticity Loss: {'Active' if metrics['plasticity_loss'] else 'Inactive'}
    """
    ax7.text(0.1, 0.5, stats_text, transform=ax7.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  Dashboard saved -> {output_path}")


def watch_mode(log_dir, interval=5):
    """实时监控模式，持续渲染图表。"""
    log_path = Path(log_dir) / "train.log"
    output_path = Path(log_dir) / "training_dashboard.png"

    print(f"[Watch Mode] Monitoring {log_path}")
    print(f"  Dashboard will be updated every {interval}s -> {output_path}")
    print("  Press Ctrl+C to stop")

    last_size = 0
    try:
        while True:
            if log_path.exists():
                current_size = log_path.stat().st_size
                if current_size != last_size:
                    metrics = parse_log_file(log_path)
                    render_training_dashboard(metrics, output_path,
                                            title=f"Training: {Path(log_dir).name}")
                    last_size = current_size
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[Watch Mode] Stopped")


def compare_experiments(exp_dirs, output_path):
    """对比多个实验的训练曲线。"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Experiment Comparison", fontsize=16, fontweight='bold')

    colors = plt.cm.tab10(np.linspace(0, 1, len(exp_dirs)))

    for idx, exp_dir in enumerate(exp_dirs):
        log_path = Path(exp_dir) / "train.log"
        name = Path(exp_dir).name
        metrics = parse_log_file(log_path)
        steps = metrics.get('steps', [])
        color = colors[idx]

        if not steps:
            continue

        # Loss
        if metrics['loss']:
            axes[0, 0].plot(steps[:len(metrics['loss'])], metrics['loss'],
                           color=color, linewidth=1.5, label=name, alpha=0.8)
        # Base loss
        if metrics['base_loss']:
            axes[0, 1].plot(steps[:len(metrics['base_loss'])], metrics['base_loss'],
                           color=color, linewidth=1.5, label=name, alpha=0.8)
        # Certainty
        if metrics['certainty']:
            axes[1, 0].plot(steps[:len(metrics['certainty'])], metrics['certainty'],
                           color=color, linewidth=1.5, label=name, alpha=0.8)
        # LR
        if metrics['lr']:
            axes[1, 1].plot(steps[:len(metrics['lr'])], metrics['lr'],
                           color=color, linewidth=1.5, label=name, alpha=0.8)

    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Base Loss')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_title('Certainty')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison saved -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize AGI-CORTEX training")
    parser.add_argument("--log_dir", type=str, default="", help="Training output directory")
    parser.add_argument("--watch", action="store_true", help="Watch mode (real-time)")
    parser.add_argument("--render", action="store_true", help="One-time render")
    parser.add_argument("--compare", nargs='+', help="Compare multiple experiments")
    parser.add_argument("--interval", type=int, default=5, help="Watch interval (seconds)")
    args = parser.parse_args()

    if args.compare:
        output = Path(args.compare[0]).parent / "comparison.png"
        compare_experiments(args.compare, output)
        return

    if args.watch and args.log_dir:
        watch_mode(args.log_dir, args.interval)
        return

    if args.render and args.log_dir:
        log_path = Path(args.log_dir) / "train.log"
        output_path = Path(args.log_dir) / "training_dashboard.png"
        metrics = parse_log_file(log_path)
        render_training_dashboard(metrics, output_path,
                                title=f"Training: {Path(args.log_dir).name}")
        return

    # 如果没有参数，尝试渲染默认位置
    if args.log_dir:
        log_path = Path(args.log_dir) / "train.log"
        output_path = Path(args.log_dir) / "training_dashboard.png"
        if log_path.exists():
            metrics = parse_log_file(log_path)
            render_training_dashboard(metrics, output_path)
        else:
            print(f"[Error] Log not found: {log_path}")
    else:
        print("Usage:")
        print("  --watch --log_dir outputs/exp1    # 实时监控")
        print("  --render --log_dir outputs/exp1   # 一次性渲染")
        print("  --compare outputs/exp1 outputs/exp2  # 对比实验")


if __name__ == "__main__":
    main()
