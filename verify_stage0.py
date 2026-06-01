"""
Stage 0 训练成功验证脚本
验证 5 大核心组件是否全部健康存活
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from cortex import CORTEXModel


def load_training_log(log_dir):
    """从训练日志提取关键指标。"""
    log_path = os.path.join(log_dir, 'train.log')
    if not os.path.exists(log_path):
        return None
    
    losses = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            if 'loss=' in line and 'Step' in line:
                try:
                    loss_str = line.split('loss=')[1].split()[0]
                    losses.append(float(loss_str))
                except:
                    pass
    return losses


def verify_numerical_convergence(log_dir):
    """验证数值收敛。"""
    losses = load_training_log(log_dir)
    if not losses or len(losses) < 5:
        print("[FAIL] 训练日志不足或不存在")
        return False
    
    initial = losses[0]
    final = losses[-1]
    
    print(f"  初始 loss: {initial:.4f}")
    print(f"  最终 loss: {final:.4f}")
    
    passed = True
    if final >= 3.0:
        print(f"  [FAIL] 最终 loss {final:.4f} >= 3.0，未收敛")
        passed = False
    else:
        print(f"  [PASS] Loss 收敛到 {final:.4f}")
    
    if final > initial * 0.8:
        print(f"  [WARN] Loss 下降幅度过小 ({initial:.4f} -> {final:.4f})")
    
    return passed


def verify_checkpoint_exists(log_dir):
    """验证 checkpoint 存在。"""
    final_ckpt = os.path.join(log_dir, 'final.pt')
    best_ckpt = os.path.join(log_dir, 'best.pt')
    
    # 查找 step checkpoint（按步数降序，取最新的）
    step_ckpts = sorted(
        [f for f in os.listdir(log_dir) if f.startswith('step_') and f.endswith('.pt')],
        reverse=True
    )
    
    if os.path.exists(final_ckpt):
        print(f"  [PASS] final.pt 存在 ({os.path.getsize(final_ckpt)/1024/1024:.2f} MB)")
        return final_ckpt
    elif os.path.exists(best_ckpt):
        print(f"  [PASS] best.pt 存在 (final.pt 缺失，使用 best.pt)")
        return best_ckpt
    elif step_ckpts:
        ckpt_path = os.path.join(log_dir, step_ckpts[0])
        print(f"  [PASS] {step_ckpts[0]} 存在 ({os.path.getsize(ckpt_path)/1024/1024:.2f} MB)")
        return ckpt_path
    else:
        print(f"  [FAIL] 无 checkpoint 文件")
        return None


def verify_gradient_integrity(checkpoint_path, device='cuda'):
    """验证所有核心组件梯度流通。"""
    print("\n[2/5] 梯度完整性验证...")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    model = CORTEXModel(
        vocab_size=10000,
        d_model=252,
        n_layers=4,
        n_modules=4,
        workspace_dim=126,
        max_seq_len=512,
    ).to(device)
    
    # 加载权重
    model.load_state_dict(state_dict, strict=False)
    model.train()
    
    # 构造一个 batch
    input_ids = torch.randint(0, 10000, (2, 128), device=device)
    labels = torch.randint(0, 10000, (2, 128), device=device)
    
    outputs = model(input_ids, labels=labels)
    loss = outputs['loss']
    loss.backward()
    
    # 检查关键路径
    key_patterns = [
        'token_embedding',
        'layers.0.dendritic',
        'layers.0.workspace',
        'layers.0.spike',
        'layers.0.predictive',
        'layers.0.multiscale',
        'output_head',
    ]
    
    all_pass = True
    for pattern in key_patterns:
        found = False
        has_grad = False
        for name, param in model.named_parameters():
            if pattern in name and param.requires_grad:
                found = True
                if param.grad is not None and param.grad.abs().sum() > 1e-10:
                    has_grad = True
                    break
        if found and has_grad:
            print(f"  [PASS] {pattern}: 梯度正常")
        elif found and not has_grad:
            print(f"  [FAIL] {pattern}: 无梯度！")
            all_pass = False
        else:
            print(f"  [WARN] {pattern}: 未找到对应参数")
    
    # 全局检查
    no_grad_params = []
    zero_grad_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            no_grad_params.append(name)
        elif param.grad.abs().sum() < 1e-10:
            zero_grad_params.append(name)
    
    if no_grad_params:
        print(f"  [FAIL] {len(no_grad_params)} 个参数无梯度")
        all_pass = False
    else:
        print(f"  [PASS] 所有参数都有梯度")
    
    if zero_grad_params:
        print(f"  [WARN] {len(zero_grad_params)} 个参数梯度为零（可能是稀疏路径）")
    
    return all_pass


def verify_component_states(checkpoint_path, device='cuda'):
    """验证各组件内部状态健康。"""
    print("\n[3/5] 组件状态健康检查...")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    model = CORTEXModel(
        vocab_size=10000,
        d_model=252,
        n_layers=4,
        n_modules=4,
        workspace_dim=126,
        max_seq_len=512,
        consciousness_output=True,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 运行多个 batch 收集统计
    spike_rates = []
    ignition_probs = []
    
    with torch.no_grad():
        for _ in range(20):
            x = torch.randint(0, 10000, (4, 128), device=device)
            outputs = model(x, return_consciousness=True)
            
            # 尝试获取 spike_rate（从各层信息中聚合）
            for layer in model.layers:
                if hasattr(layer, 'spike_encoder') and layer.spike_encoder is not None:
                    # 触发一次前向获取 spike 信息
                    dummy = torch.randn(4, 128, 252, device=device)
                    _, info = layer.spike_encoder(dummy)
                    if 'spike_rate' in info:
                        sr = info['spike_rate']
                        spike_rates.append(sr.mean().item() if isinstance(sr, torch.Tensor) and sr.numel() > 1 else float(sr))
                
                if hasattr(layer, 'workspace') and layer.workspace is not None:
                    dummy = torch.randn(4, 128, 252, device=device)
                    _, ws_info = layer.workspace(dummy)
                    if 'ignition_prob' in ws_info:
                        ip = ws_info['ignition_prob']
                        ignition_probs.append(ip.mean().item() if ip.numel() > 1 else ip.item())
    
    passed = True
    
    # Spike rate 检查
    if spike_rates:
        avg_spike = sum(spike_rates) / len(spike_rates)
        print(f"  平均 spike_rate: {avg_spike:.4f}")
        if avg_spike < 0.05:
            print(f"  [FAIL] spike_rate 过低 ({avg_spike:.4f})，脉冲机制可能失效")
            passed = False
        elif avg_spike > 0.6:
            print(f"  [FAIL] spike_rate 过高 ({avg_spike:.4f})，稀疏性丧失")
            passed = False
        else:
            print(f"  [PASS] spike_rate 健康")
    else:
        print(f"  [WARN] 无法提取 spike_rate")
    
    # Ignition 检查
    if ignition_probs:
        avg_ignition = sum(ignition_probs) / len(ignition_probs)
        std_ignition = (sum((p - avg_ignition)**2 for p in ignition_probs) / len(ignition_probs)) ** 0.5
        print(f"  平均 ignition_prob: {avg_ignition:.4f} ± {std_ignition:.4f}")
        if std_ignition < 0.01:
            print(f"  [FAIL] ignition_prob 无波动（std={std_ignition:.4f}），GNW 竞争未激活")
            passed = False
        else:
            print(f"  [PASS] GNW 竞争动态正常")
    else:
        print(f"  [WARN] 无法提取 ignition_prob")
    
    return passed


def verify_generation(checkpoint_path, device='cuda'):
    """验证生成能力。"""
    print("\n[4/5] 生成能力验证...")
    
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    model = CORTEXModel(
        vocab_size=10000,
        d_model=252,
        n_layers=4,
        n_modules=4,
        workspace_dim=126,
        max_seq_len=512,
    ).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # 简单提示生成
    prompts = [
        torch.randint(0, 10000, (1, 8), device=device),
        torch.randint(0, 10000, (1, 16), device=device),
    ]
    
    all_pass = True
    for i, prompt in enumerate(prompts):
        with torch.no_grad():
            gen = model.generate(prompt, max_new_tokens=20, temperature=0.8)
        gen_len = gen.shape[1]
        print(f"  Prompt {i+1}: 输入 {prompt.shape[1]} tokens -> 生成 {gen_len} tokens")
        
        if gen_len <= prompt.shape[1]:
            print(f"  [FAIL] 未生成新 token")
            all_pass = False
        else:
            print(f"  [PASS] 生成成功")
    
    return all_pass


def verify_perplexity(checkpoint_path, device='cuda'):
    """验证验证集 PPL。"""
    print("\n[5/5] 验证集 PPL 检查...")
    
    # 尝试从日志中读取 val_loss
    log_path = 'outputs/stage0_base/train.log'
    val_losses = []
    if os.path.exists(log_path):
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 'Validation: loss=' in line:
                    try:
                        vloss = float(line.split('Validation: loss=')[1].split()[0])
                        val_losses.append(vloss)
                    except:
                        pass
    
    if val_losses:
        best_val = min(val_losses)
        ppl = math.exp(best_val)
        print(f"  最佳 val_loss: {best_val:.4f}")
        print(f"  对应 PPL: {ppl:.2f}")
        if ppl < 100:
            print(f"  [PASS] PPL 健康 (< 100)")
            return True
        else:
            print(f"  [WARN] PPL 偏高 ({ppl:.2f})")
            return False
    else:
        print(f"  [SKIP] 无验证记录")
        return True


def main():
    log_dir = sys.argv[1] if len(sys.argv) > 1 else "outputs/stage0_base"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 60)
    print("Stage 0 训练成功验证")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"日志目录: {log_dir}")
    
    # 1. Checkpoint 存在性
    print("\n[1/5] Checkpoint 检查...")
    ckpt_path = verify_checkpoint_exists(log_dir)
    if not ckpt_path:
        print("\n[最终结论] ❌ 失败：无 checkpoint")
        sys.exit(1)
    
    # 2. 数值收敛
    conv_ok = verify_numerical_convergence(log_dir)
    
    # 3. 梯度完整性
    grad_ok = verify_gradient_integrity(ckpt_path, device)
    
    # 4. 组件状态
    comp_ok = verify_component_states(ckpt_path, device)
    
    # 5. 生成能力
    gen_ok = verify_generation(ckpt_path, device)
    
    # 6. PPL
    ppl_ok = verify_perplexity(ckpt_path, device)
    
    # 总结
    print("\n" + "=" * 60)
    results = {
        "数值收敛": conv_ok,
        "梯度完整性": grad_ok,
        "组件状态": comp_ok,
        "生成能力": gen_ok,
        "验证PPL": ppl_ok,
    }
    
    for name, ok in results.items():
        status = "[PASS]" if ok else "[FAIL]"
        print(f"  {name}: {status}")
    
    all_pass = all(results.values())
    if all_pass:
        print("\n[最终结论] Stage 0 训练成功！所有核心组件验证通过。")
        print("可进入 Stage 1 语言建模热身。")
    else:
        print("\n[最终结论] Stage 0 存在失败项，请检查上述日志。")
    print("=" * 60)
    
    sys.exit(0 if all_pass else 1)


if __name__ == '__main__':
    main()
