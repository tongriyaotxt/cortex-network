"""
AGI-CORTEX 一键训练管道

完整流程：数据生成 → 分阶段训练 → 实时可视化 → 对比报告

Usage:
    # 完整流程（所有阶段）
    python scripts/train_pipeline.py --full --viz

    # 仅生成数据
    python scripts/train_pipeline.py --data_only

    # 从指定阶段开始
    python scripts/train_pipeline.py --from_stage 2 --viz

    # 快速验证（小数据 + 少步骤）
    python scripts/train_pipeline.py --quick --viz
"""

import os
import sys
import json
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 项目路径
PROJECT_ROOT = Path(__file__).parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONFIGS_DIR = PROJECT_ROOT / "configs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 阶段配置
STAGES = [
    {
        "name": "stage0_base",
        "config": "configs/stage0_base.yaml",
        "desc": "基础架构热身（纯 CORTEX，无 AGI 模块）",
        "data": "data/corpus/train.txt",
    },
    {
        "name": "stage1_lm",
        "config": "configs/stage1_lm.yaml",
        "desc": "语言建模训练（启用脉冲，无 AGI 模块）",
        "data": "data/corpus/train.txt",
    },
    {
        "name": "stage2_reasoning",
        "config": "configs/stage2_reasoning.yaml",
        "desc": "符号推理 + 元认知（M1 + M2）",
        "data": "data/reasoning/train.jsonl",
    },
    {
        "name": "stage3_rl",
        "config": "configs/stage3_rl.yaml",
        "desc": "具身交互 + 层次规划（M3 + M4）",
        "data": "data/corpus/train.txt",
    },
    {
        "name": "stage4_full",
        "config": "configs/stage4_full.yaml",
        "desc": "全模块联合训练（M1-M6）",
        "data": "data/corpus/train.txt",
    },
]


def run_command(cmd, desc=""):
    """运行命令并实时输出。"""
    if desc:
        print(f"\n{'='*60}")
        print(f"[RUN] {desc}")
        print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    start = time.time()
    result = subprocess.run(cmd, capture_output=False, text=True)
    elapsed = time.time() - start
    
    if result.returncode != 0:
        print(f"[ERROR] Command failed with code {result.returncode}")
        return False, elapsed
    
    print(f"[OK] Completed in {elapsed:.1f}s")
    return True, elapsed


def generate_data(args):
    """生成训练数据。"""
    print("\n" + "="*60)
    print("STEP 1: 数据生成")
    print("="*60)
    
    cmd = [sys.executable, str(SCRIPTS_DIR / "generate_enriched_data.py"), "--all", "--viz"]
    
    if args.quick:
        cmd.extend(["--n_corpus", "5000", "--n_reasoning", "1000", 
                    "--n_metacog", "500", "--n_rl", "200"])
    
    success, _ = run_command(cmd, "生成全部训练数据")
    return success


def train_stage(stage_info, args, resume_from=None):
    """训练单个阶段。"""
    config_path = PROJECT_ROOT / stage_info["config"]
    output_dir = OUTPUTS_DIR / stage_info["name"]
    
    if not config_path.exists():
        print(f"[Warning] Config not found: {config_path}, using default args")
        # 使用默认参数
        cmd = [
            sys.executable, str(EXAMPLES_DIR / "train_agi_cortex.py"),
            "--data_path", stage_info["data"],
            "--output_dir", str(output_dir),
            "--total_steps", str(1000 if args.quick else 10000),
            "--batch_size", "8",
            "--seq_len", "128",
            "--log_interval", "100",
            "--val_interval", "500",
            "--save_interval", "1000",
            "--device", args.device,
        ]
    else:
        cmd = [
            sys.executable, str(EXAMPLES_DIR / "train_agi_cortex.py"),
            "--config", str(config_path),
            "--output_dir", str(output_dir),
            "--device", args.device,
        ]
    
    if resume_from:
        cmd.extend(["--resume", str(resume_from)])
    
    if args.quick:
        # 快速模式覆盖 total_steps
        cmd.extend(["--total_steps", "1000"])
    
    success, elapsed = run_command(cmd, stage_info["desc"])
    
    # 返回 best checkpoint 路径
    best_ckpt = output_dir / "best.pt"
    if best_ckpt.exists():
        return success, str(best_ckpt), elapsed
    final_ckpt = output_dir / "final.pt"
    if final_ckpt.exists():
        return success, str(final_ckpt), elapsed
    return success, None, elapsed


def visualize_stage(stage_name, args):
    """可视化单个阶段的训练结果。"""
    log_dir = OUTPUTS_DIR / stage_name
    viz_script = SCRIPTS_DIR / "visualize_training.py"
    
    cmd = [
        sys.executable, str(viz_script),
        "--render",
        "--log_dir", str(log_dir),
    ]
    success, _ = run_command(cmd, f"可视化 {stage_name}")
    return success


def generate_summary_report(results, output_path):
    """生成训练总结报告。"""
    report = {
        "timestamp": datetime.now().isoformat(),
        "stages": []
    }
    
    total_time = 0
    for r in results:
        report["stages"].append({
            "name": r["name"],
            "desc": r["desc"],
            "success": r["success"],
            "checkpoint": r.get("checkpoint"),
            "elapsed_seconds": r.get("elapsed", 0),
        })
        total_time += r.get("elapsed", 0)
    
    report["total_time_seconds"] = total_time
    report["total_time_hours"] = total_time / 3600
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print("训练总结报告")
    print(f"{'='*60}")
    print(f"总时间: {total_time/3600:.2f} 小时")
    print(f"\n各阶段结果:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {r['name']}: {r['desc']} ({r.get('elapsed', 0)/60:.1f} min)")
    print(f"\n报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="AGI-CORTEX Training Pipeline")
    parser.add_argument("--full", action="store_true", help="Run all stages")
    parser.add_argument("--from_stage", type=int, default=0, help="Start from stage (0-4)")
    parser.add_argument("--to_stage", type=int, default=4, help="End at stage (0-4)")
    parser.add_argument("--data_only", action="store_true", help="Only generate data")
    parser.add_argument("--quick", action="store_true", help="Quick mode (small data, few steps)")
    parser.add_argument("--viz", action="store_true", help="Generate visualizations")
    parser.add_argument("--device", type=str, default="cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu",
                       help="Training device")
    parser.add_argument("--skip_data", action="store_true", help="Skip data generation")
    args = parser.parse_args()

    print("="*60)
    print("AGI-CORTEX Training Pipeline")
    print("="*60)
    print(f"Device: {args.device}")
    print(f"Quick mode: {args.quick}")
    print(f"Visualization: {args.viz}")
    
    # 步骤 1：数据生成
    if not args.skip_data:
        if not generate_data(args):
            print("[ERROR] 数据生成失败，终止")
            return
    else:
        print("\n[SKIP] 跳过数据生成")
    
    if args.data_only:
        print("\n[Done] 仅生成数据，退出")
        return
    
    # 步骤 2-6：分阶段训练
    results = []
    last_checkpoint = None
    
    stages_to_run = STAGES[args.from_stage:args.to_stage+1]
    
    for stage in stages_to_run:
        print(f"\n{'='*60}")
        print(f"Stage: {stage['name']}")
        print(f"Description: {stage['desc']}")
        print(f"{'='*60}")
        
        success, ckpt, elapsed = train_stage(stage, args, resume_from=last_checkpoint)
        
        results.append({
            "name": stage["name"],
            "desc": stage["desc"],
            "success": success,
            "checkpoint": ckpt,
            "elapsed": elapsed,
        })
        
        if not success:
            print(f"[WARNING] {stage['name']} 训练失败，继续下一阶段...")
        
        if ckpt:
            last_checkpoint = ckpt
        
        # 可视化
        if args.viz:
            visualize_stage(stage["name"], args)
    
    # 步骤 7：总结报告
    report_path = OUTPUTS_DIR / "training_summary.json"
    generate_summary_report(results, report_path)
    
    # 对比可视化
    if args.viz and len(results) > 1:
        exp_dirs = [str(OUTPUTS_DIR / r["name"]) for r in results if r["success"]]
        if len(exp_dirs) > 1:
            viz_script = SCRIPTS_DIR / "visualize_training.py"
            cmd = [sys.executable, str(viz_script), "--compare"] + exp_dirs
            run_command(cmd, "实验对比")
    
    print("\n" + "="*60)
    print("训练管道完成!")
    print(f"输出目录: {OUTPUTS_DIR.absolute()}")
    print("="*60)


if __name__ == "__main__":
    main()
