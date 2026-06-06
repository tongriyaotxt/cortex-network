"""
Stage 0 模型评估脚本：验证基础 CORTEX 是否真正学到了东西
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import math
import argparse

from cortex import CORTEXModel


def char_tokenize(text, vocab_size=256):
    return [ord(c) % vocab_size for c in text]


def load_model_from_ckpt(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', ckpt.get('args', {}))
    
    model = CORTEXModel(
        vocab_size=config.get('vocab_size', 256),
        d_model=config.get('d_model', 252),
        n_layers=config.get('n_layers', 4),
        n_modules=config.get('n_modules', 4),
        workspace_dim=config.get('workspace_dim', 126),
        n_branches=config.get('n_branches', 4),
        n_timescales=config.get('n_timescales', 3),
        max_seq_len=config.get('max_seq_len', 512),
        dropout=0.0,
        consciousness_output=True,
        causal=True,
    )
    state_dict = ckpt.get('model_state_dict', ckpt.get('model', None))
    if state_dict is None:
        raise ValueError("Checkpoint does not contain model state dict")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, config


def eval_perplexity(model, text, seq_len=128, device='cuda'):
    tokens = char_tokenize(text)
    total_loss = 0.0
    total_tokens = 0
    
    model.eval()
    with torch.no_grad():
        for i in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens[i:i + seq_len + 1]
            if len(chunk) < seq_len + 1:
                continue
            inputs = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            labels = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
            outputs = model(inputs, labels=labels)
            n = labels.numel()
            total_loss += outputs['loss'].item() * n
            total_tokens += n
    
    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl


def eval_generation(model, prompts, device='cuda', max_new=40, temperature=0.8):
    results = []
    with torch.no_grad():
        for p in prompts:
            tokens = char_tokenize(p)
            inp = torch.tensor([tokens], dtype=torch.long, device=device)
            out = model.generate(inp, max_new_tokens=max_new, temperature=temperature, top_k=40)
            gen = out[0].cpu().tolist()[len(tokens):]
            text = ''.join(chr(c) if 32 <= c < 127 else '?' for c in gen)
            results.append((p, text))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--val_path', type=str, default='data/corpus/val.txt')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}")
        return
    
    # 加载验证文本
    if os.path.exists(args.val_path):
        with open(args.val_path, 'r', encoding='utf-8') as f:
            val_text = f.read()
    else:
        val_text = "Alice hid the sword in the tower. Eve sought it.\nBob gave the lamp to Alice. Alice was scared."
    
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Val text length: {len(val_text)} chars")
    print(f"Device: {args.device}")
    
    # 加载训练后的模型
    print("\n[1] Loading trained model...")
    trained, config = load_model_from_ckpt(args.checkpoint, args.device)
    print(f"    Model params: {sum(p.numel() for p in trained.parameters()):,}")
    
    # 计算训练后模型的困惑度
    print("\n[2] Computing perplexity (trained)...")
    loss_trained, ppl_trained = eval_perplexity(trained, val_text, device=args.device)
    print(f"    Loss: {loss_trained:.4f} | PPL: {ppl_trained:.2f}")
    
    # 创建随机模型做对比
    print("\n[3] Computing perplexity (untrained/random)...")
    untrained = CORTEXModel(
        vocab_size=config.get('vocab_size', 256),
        d_model=config.get('d_model', 252),
        n_layers=config.get('n_layers', 4),
        n_modules=config.get('n_modules', 4),
        workspace_dim=config.get('workspace_dim', 126),
        n_branches=config.get('n_branches', 4),
        n_timescales=config.get('n_timescales', 3),
        max_seq_len=config.get('max_seq_len', 512),
        dropout=0.0,
        consciousness_output=True,
        causal=True,
    ).to(args.device)
    untrained.eval()
    loss_rand, ppl_rand = eval_perplexity(untrained, val_text, device=args.device)
    print(f"    Loss: {loss_rand:.4f} | PPL: {ppl_rand:.2f}")
    
    # 对比结果
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"Random model:     PPL = {ppl_rand:.2f}")
    print(f"Trained model:    PPL = {ppl_trained:.2f}")
    if ppl_trained < ppl_rand:
        print(f"Improvement:      PPL reduced by {ppl_rand/ppl_trained:.2f}x")
        print("VERDICT: Model has learned something!")
    else:
        print("WARNING: Trained model is NOT better than random!")
    print("="*50)
    
    # 生成样本
    print("\n[4] Generation samples (trained model):")
    prompts = ["Alice ", "Bob gave ", "The ", "In the ", "Eve "]
    gens = eval_generation(trained, prompts, device=args.device, max_new=40, temperature=0.8)
    for prompt, text in gens:
        print(f"  '{prompt}' -> '{text}'")
    
    print("\n[5] Generation samples (random model):")
    gens_rand = eval_generation(untrained, prompts[:3], device=args.device, max_new=40, temperature=0.8)
    for prompt, text in gens_rand:
        print(f"  '{prompt}' -> '{text}'")


if __name__ == '__main__':
    main()
