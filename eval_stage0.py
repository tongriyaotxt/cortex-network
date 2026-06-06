"""Evaluate Stage 0 CORTEX baseline on language modeling."""
import torch
import os
import json
import math
import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from examples.train_agi_cortex import SimpleTokenizer, TokenDataset, lm_collate_fn
from torch.utils.data import DataLoader
from cortex.agi_cortex_model import AGICORTEXModel


def compute_perplexity(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, labels, attention_mask = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            attention_mask = attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            bs = input_ids.size(0)
            total_loss += outputs['loss'].item() * bs
            total_samples += bs
    avg_nll = total_loss / total_samples
    ppl = math.exp(avg_nll)
    return ppl, avg_nll


def generate_samples(model, tokenizer, prompts, max_new_tokens=200, temperature=0.8, top_k=40):
    model.eval()
    # Move model to CPU for generation to avoid async CUDA assert issues
    cpu_model = model.to('cpu')
    results = []
    for prompt_text in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt_text, max_length=512)])
        with torch.no_grad():
            output = cpu_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )
        gen_text = tokenizer.decode(output[0].tolist())
        results.append({
            'prompt': prompt_text,
            'generated': gen_text,
        })
    # Move back to original device
    model.to(next(model.parameters()).device)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='outputs/stage0_full/final.pt')
    parser.add_argument('--config', type=str, default='configs/stage0_base.yaml')
    parser.add_argument('--data_path', type=str, default='data/corpus/train.txt')
    parser.add_argument('--val_path', type=str, default='data/corpus/val.txt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_new_tokens', type=int, default=200)
    args = parser.parse_args()

    import yaml

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Build model (same class as training script)
    model = AGICORTEXModel(
        vocab_size=cfg['vocab_size'],
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        n_modules=cfg.get('n_modules', 4),
        workspace_dim=cfg.get('workspace_dim', 128),
        n_branches=cfg.get('n_branches', 4),
        n_timescales=cfg.get('n_timescales', 3),
        max_seq_len=cfg.get('max_seq_len', 512),
        dropout=cfg.get('dropout', 0.1),
        causal=True,
        tie_weights=False,  # MUST match training config
        consciousness_output=True,
        # AGI modules all off for stage 0
        use_symbolic=False,
        use_self_modeling=False,
        use_embodied=False,
        use_hierarchical=False,
        use_continual=False,
        use_causal=False,
    )
    model.to(device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    step = ckpt.get('global_step', 'unknown')
    print(f"Loaded checkpoint from step {step}")

    tokenizer = SimpleTokenizer(vocab_size=cfg['vocab_size'])

    # Perplexity on train-val split (same as training validation)
    if args.data_path and os.path.exists(args.data_path):
        with open(args.data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = tokenizer.encode(text, max_length=None, truncation=False)
        split = int(len(tokens) * 0.95)
        val_tokens = tokens[split:]
        val_dataset = TokenDataset(val_tokens, seq_len=cfg.get('seq_len', 128))
        val_loader = DataLoader(val_dataset, batch_size=cfg.get('batch_size', 16), shuffle=False, collate_fn=lm_collate_fn)
        ppl, nll = compute_perplexity(model, val_loader, device)
        print(f"\n=== Train-val Perplexity: {ppl:.2f} | NLL: {nll:.4f} ===")
    else:
        print("No train data found, skipping train-val perplexity.")

    # Perplexity on separate val file
    if args.val_path and os.path.exists(args.val_path):
        with open(args.val_path, 'r', encoding='utf-8') as f:
            val_text = f.read()
        val_tokens = tokenizer.encode(val_text, max_length=None, truncation=False)
        val_dataset = TokenDataset(val_tokens, seq_len=cfg.get('seq_len', 128))
        val_loader = DataLoader(val_dataset, batch_size=cfg.get('batch_size', 16), shuffle=False, collate_fn=lm_collate_fn)
        ppl, nll = compute_perplexity(model, val_loader, device)
        print(f"\n=== Separate-val Perplexity: {ppl:.2f} | NLL: {nll:.4f} ===")
    else:
        print("No separate val data found.")

    # Generation samples (limit to 120 tokens to avoid NaN instability in long sequences)
    gen_length = min(args.max_new_tokens, 120)
    prompts = [
        "The architecture of mind",
        "Consciousness emerges",
        "Neural networks process",
        "The human brain",
        "Predictive coding",
    ]
    print(f"\n=== Generation Samples (max {gen_length} tokens) ===")
    samples = generate_samples(model, tokenizer, prompts, max_new_tokens=gen_length)
    for s in samples:
        print(f"\nPrompt: {s['prompt']}")
        print(f"Generated: {s['generated']}")
        print("-" * 60)


if __name__ == '__main__':
    main()
