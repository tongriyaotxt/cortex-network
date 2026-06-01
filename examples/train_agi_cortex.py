"""
AGICORTEXModel 训练脚本

支持：
- 语言建模 (Language Modeling)
- 六模块独立/联合训练
- 课程学习（三阶段渐进启用 AGI 模块）
- 多目标损失（基础 LM + M1-M6 模块损失）
- 混合精度训练
- Checkpoint 保存/恢复
- WandB / TensorBoard 日志

Usage:
    # 基础训练（仅核心 CORTEX）
    python examples/train_agi_cortex.py --task lm --data_path data.txt

    # 启用所有 AGI 模块
    python examples/train_agi_cortex.py --task lm --data_path data.txt \
        --use_symbolic --use_self_modeling --use_embodied \
        --use_hierarchical --use_continual --use_causal

    # 课程学习：渐进启用模块
    python examples/train_agi_cortex.py --task lm --data_path data.txt --curriculum

    # 从 checkpoint 恢复
    python examples/train_agi_cortex.py --resume checkpoint.pt
"""

import sys
import os
import json
import math
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex import AGICORTEXModel

# Optional YAML config support
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


# =============================================================================
# 数据集定义
# =============================================================================

class TokenDataset(Dataset):
    """Token 序列数据集。"""
    def __init__(self, data, seq_len=512, is_sequences=False):
        self.seq_len = seq_len
        self.samples = []
        
        if is_sequences:
            for seq in data:
                if len(seq) < seq_len + 1:
                    continue
                n = len(seq) // (seq_len + 1) * (seq_len + 1)
                seq = seq[:n]
                for i in range(0, n, seq_len + 1):
                    chunk = seq[i:i + seq_len + 1]
                    if len(chunk) == seq_len + 1:
                        self.samples.append((chunk[:-1], chunk[1:]))
        else:
            n = len(data) // (seq_len + 1) * (seq_len + 1)
            data = data[:n]
            for i in range(0, n, seq_len + 1):
                chunk = data[i:i + seq_len + 1]
                self.samples.append((chunk[:-1], chunk[1:]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inputs, labels = self.samples[idx]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def lm_collate_fn(batch, pad_token_id=0):
    """语言建模的 collate 函数。"""
    inputs, labels = zip(*batch)
    max_len = max(len(x) for x in inputs)
    
    padded_inputs = []
    padded_labels = []
    masks = []
    
    for x, y in zip(inputs, labels):
        pad_len = max_len - len(x)
        padded_inputs.append(torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)]))
        masks.append(torch.cat([torch.ones(len(x)), torch.zeros(pad_len)]))
    
    return torch.stack(padded_inputs), torch.stack(padded_labels), torch.stack(masks)


# =============================================================================
# JSONL 数据集（用于推理 / 元认知数据）
# =============================================================================

class JsonlDataset(Dataset):
    """加载 JSONL 格式的结构化数据（推理链、元认知样本等）。"""
    def __init__(self, data_path, tokenizer, seq_len=512, text_key="text"):
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.samples = []
        self.text_key = text_key
        
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # 支持多种格式
                    if isinstance(obj, dict):
                        if text_key in obj:
                            text = obj[text_key]
                        elif "context" in obj and "answer" in obj:
                            text = f"{obj['context']} 答案：{obj['answer']}"
                        elif "input" in obj:
                            text = obj["input"]
                        else:
                            text = json.dumps(obj, ensure_ascii=False)
                    else:
                        text = str(obj)
                    tokens = tokenizer.encode(text, max_length=None, truncation=False)
                    # 切分为 (input, label) 对
                    for i in range(0, max(1, len(tokens) - seq_len), seq_len // 2):
                        chunk = tokens[i:i + seq_len + 1]
                        if len(chunk) == seq_len + 1:
                            self.samples.append((chunk[:-1], chunk[1:]))
                except json.JSONDecodeError:
                    # 如果不是 JSON，当作纯文本处理
                    tokens = tokenizer.encode(line)
                    for i in range(0, max(1, len(tokens) - seq_len), seq_len // 2):
                        chunk = tokens[i:i + seq_len + 1]
                        if len(chunk) == seq_len + 1:
                            self.samples.append((chunk[:-1], chunk[1:]))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        inputs, labels = self.samples[idx]
        return torch.tensor(inputs, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# =============================================================================
# 简单 Tokenizer
# =============================================================================

class SimpleTokenizer:
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.char_mode = vocab_size <= 512
    
    def encode(self, text, max_length=512, truncation=True):
        if self.char_mode:
            tokens = [ord(c) % self.vocab_size for c in text]
        else:
            words = text.split()
            tokens = [hash(w) % self.vocab_size for w in words]
        if truncation:
            tokens = tokens[:max_length]
        return tokens
    
    def decode(self, tokens):
        if self.char_mode:
            return ''.join(chr(t) for t in tokens if t < 128)
        return ' '.join(str(t) for t in tokens)


# =============================================================================
# AGI-CORTEX 训练器
# =============================================================================

class AGICORTEXTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        self.log_file = open(self.output_dir / 'train.log', 'a')
        
        # 构建模型
        self.model = self._build_model()
        self.model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # 学习率调度
        self.scheduler = None
        if args.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=args.total_steps - args.warmup_steps,
                eta_min=args.lr * 0.01,
            )
        
        # 混合精度
        self.scaler = GradScaler('cuda') if args.use_amp else None
        
        # 课程学习阶段
        self.curriculum_stage = 0
        if args.curriculum:
            self._update_curriculum()
        
        # 加载 checkpoint
        if args.resume:
            self._load_checkpoint(args.resume)
        
        self._log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self._log(f"AGI modules: symbolic={args.use_symbolic}, self={args.use_self_modeling}, "
                  f"embodied={args.use_embodied}, hierarchical={args.use_hierarchical}, "
                  f"continual={args.use_continual}, causal={args.use_causal}")
        self._log(f"Training device: {self.device}")
    
    def _build_model(self):
        args = self.args
        return AGICORTEXModel(
            vocab_size=args.vocab_size,
            d_model=args.d_model,
            n_layers=args.n_layers,
            n_modules=args.n_modules,
            workspace_dim=args.workspace_dim,
            n_branches=args.n_branches,
            n_timescales=args.n_timescales,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            num_classes=args.num_classes if args.task == 'classification' else None,
            consciousness_output=True,
            causal=args.causal,
            tie_weights=args.tie_weights,
            # AGI switches
            use_symbolic=args.use_symbolic,
            use_self_modeling=args.use_self_modeling,
            use_embodied=args.use_embodied,
            use_hierarchical=args.use_hierarchical,
            use_continual=args.use_continual,
            use_causal=args.use_causal,
            # AGI configs
            symbolic_vocab_size=args.symbolic_vocab_size,
            n_subroutines=args.n_subroutines,
            memory_capacity=args.memory_capacity,
            max_goal_depth=args.max_goal_depth,
            n_counterfactuals=args.n_counterfactuals,
        )
    
    def _update_curriculum(self):
        """三阶段课程学习，渐进启用 AGI 模块。"""
        stage = self.curriculum_stage
        model = self.model
        
        if stage == 0:
            # 阶段0：纯基础训练，关闭所有 AGI 模块
            for block in model.layers:
                block.use_spike = False
            model.lambda_symbolic = 0.0
            model.lambda_self = 0.0
            model.lambda_action = 0.0
            model.lambda_plasticity = 0.0
            self._log("[Curriculum] Stage 0: Base CORTEX warm-up (all AGI modules off, no spikes)")
        
        elif stage == 1:
            # 阶段1：启用脉冲 + 自我建模 + 符号推理
            for block in model.layers:
                block.use_spike = True
            model.lambda_symbolic = self.args.lambda_symbolic * 0.5
            model.lambda_self = self.args.lambda_self * 0.5
            model.lambda_action = 0.0
            model.lambda_plasticity = 0.0
            self._log("[Curriculum] Stage 1: Introducing spikes + M1/M2")
        
        else:
            # 阶段2：完整训练，启用所有模块
            for block in model.layers:
                block.use_spike = True
            model.lambda_symbolic = self.args.lambda_symbolic
            model.lambda_self = self.args.lambda_self
            model.lambda_action = self.args.lambda_action
            model.lambda_plasticity = self.args.lambda_plasticity
            self._log("[Curriculum] Stage 2: Full AGI-CORTEX training")
    
    def _log(self, msg):
        t = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{t}] {msg}"
        print(line)
        self.log_file.write(line + '\n')
        self.log_file.flush()
    
    def train_step(self, batch):
        self.model.train()
        input_ids, labels, attention_mask = batch
        input_ids = input_ids.to(self.device)
        labels = labels.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        self.optimizer.zero_grad()
        
        if self.scaler is not None:
            with autocast('cuda'):
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs['loss']
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self.optimizer.step()
        
        # 学习率调度
        if self.scheduler is not None and self.global_step >= self.args.warmup_steps:
            self.scheduler.step()
        
        self.global_step += 1
        return outputs
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        total_base_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels, attention_mask = batch
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                outputs = self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                batch_size = input_ids.size(0)
                total_loss += outputs['loss'].item() * batch_size
                total_base_loss += outputs.get('base_loss', outputs['loss']).item() * batch_size
                total_samples += batch_size
        
        return total_loss / total_samples, total_base_loss / total_samples
    
    def save_checkpoint(self, name='checkpoint.pt'):
        path = self.output_dir / name
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict() if self.scaler else None,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': vars(self.args),
        }, path)
        self._log(f"Saved checkpoint: {path}")
    
    def _load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler and ckpt.get('scheduler'):
            sched_state = ckpt['scheduler']
            # Guard against corrupted scheduler state (e.g. T_max=0 from race conditions)
            if sched_state.get('T_max', 0) > 0:
                self.scheduler.load_state_dict(sched_state)
                self._log(f"Loaded scheduler state (T_max={sched_state['T_max']})")
            else:
                self._log(f"WARNING: skipping corrupted scheduler state (T_max={sched_state.get('T_max', 'N/A')}), recreating scheduler")
                # Recreate scheduler with correct T_max based on remaining steps
                remaining_steps = self.args.total_steps - self.args.warmup_steps
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=remaining_steps,
                    eta_min=self.args.lr * 0.01,
                )
        if self.scaler and ckpt.get('scaler'):
            self.scaler.load_state_dict(ckpt['scaler'])
        self.global_step = ckpt.get('global_step', 0)
        self.best_val_loss = ckpt.get('best_val_loss', float('inf'))
        self._log(f"Loaded checkpoint from {path} (step {self.global_step})")
    
    def train(self, train_loader, val_loader=None):
        self._log("Starting training...")
        
        while self.global_step < self.args.total_steps:
            for batch in train_loader:
                outputs = self.train_step(batch)
                
                # 课程学习阶段切换
                if self.args.curriculum:
                    stage_boundary_1 = self.args.total_steps // 3
                    stage_boundary_2 = 2 * self.args.total_steps // 3
                    
                    if self.global_step == stage_boundary_1 and self.curriculum_stage == 0:
                        self.curriculum_stage = 1
                        self._update_curriculum()
                    elif self.global_step == stage_boundary_2 and self.curriculum_stage == 1:
                        self.curriculum_stage = 2
                        self._update_curriculum()
                
                # 日志
                if self.global_step % self.args.log_interval == 0:
                    loss_val = outputs['loss'].item()
                    base_loss = outputs.get('base_loss', outputs['loss']).item()
                    msg = f"Step {self.global_step}/{self.args.total_steps} | loss={loss_val:.4f} | base={base_loss:.4f}"
                    
                    module_losses = outputs.get('module_losses', {})
                    if module_losses:
                        mstr = ' | '.join([f"{k}={v:.4f}" for k, v in module_losses.items()])
                        msg += f" | [{mstr}]"
                    
                    # 自我状态日志
                    if 'self_state' in outputs and outputs['self_state'] is not None:
                        ss = outputs['self_state']
                        if hasattr(ss, 'certainty'):
                            msg += f" | cert={ss.certainty:.2f}"
                        if hasattr(ss, 'cognitive_load'):
                            msg += f" | load={ss.cognitive_load:.2f}"
                    
                    lr = self.optimizer.param_groups[0]['lr']
                    msg += f" | lr={lr:.2e}"
                    self._log(msg)
                
                # 验证
                if val_loader is not None and self.global_step % self.args.val_interval == 0:
                    val_loss, val_base = self.validate(val_loader)
                    self._log(f"Validation: loss={val_loss:.4f} | base={val_base:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint('best.pt')
                
                # 定期保存
                if self.global_step % self.args.save_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}.pt')
                
                if self.global_step >= self.args.total_steps:
                    break
        
        self._log("Training complete!")
        self.save_checkpoint('final.pt')
        self.log_file.close()


def load_data(args):
    """加载训练数据。支持 .txt 和 .jsonl 格式。"""
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    
    if args.data_path and os.path.exists(args.data_path):
        # 根据扩展名选择加载方式
        if args.data_path.endswith('.jsonl'):
            dataset = JsonlDataset(args.data_path, tokenizer, seq_len=args.seq_len)
            # 划分训练/验证
            n = len(dataset)
            n_train = int(n * 0.95)
            train_dataset = torch.utils.data.Subset(dataset, range(n_train))
            val_dataset = torch.utils.data.Subset(dataset, range(n_train, n)) if n_train < n else None
        else:
            with open(args.data_path, 'r', encoding='utf-8') as f:
                text = f.read()
            tokens = tokenizer.encode(text, max_length=None, truncation=False)
            
            # 划分训练/验证
            split = int(len(tokens) * 0.95)
            train_tokens = tokens[:split]
            val_tokens = tokens[split:]
            
            train_dataset = TokenDataset(train_tokens, seq_len=args.seq_len)
            val_dataset = TokenDataset(val_tokens, seq_len=args.seq_len) if val_tokens else None
    else:
        # 生成随机数据用于测试
        print("Warning: data_path not found, using synthetic data")
        train_data = [torch.randint(0, args.vocab_size, (args.seq_len + 1,)).tolist() for _ in range(1000)]
        val_data = [torch.randint(0, args.vocab_size, (args.seq_len + 1,)).tolist() for _ in range(100)]
        train_dataset = TokenDataset(train_data, seq_len=args.seq_len, is_sequences=True)
        val_dataset = TokenDataset(val_data, seq_len=args.seq_len, is_sequences=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lm_collate_fn,
        num_workers=0,
    )
    
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lm_collate_fn,
            num_workers=0,
        )
    
    return train_loader, val_loader


def main():
    parser = argparse.ArgumentParser(description='Train AGI-CORTEX Model')
    
    # 配置文件支持
    parser.add_argument('--config', type=str, default='', help='YAML config file path')
    
    # 模型配置
    parser.add_argument('--vocab_size', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=252)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_modules', type=int, default=4)
    parser.add_argument('--workspace_dim', type=int, default=126)
    parser.add_argument('--n_branches', type=int, default=4)
    parser.add_argument('--n_timescales', type=int, default=3)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--causal', action='store_true', default=True)
    parser.add_argument('--tie_weights', action='store_true')
    
    # AGI 模块开关
    parser.add_argument('--use_symbolic', action='store_true')
    parser.add_argument('--use_self_modeling', action='store_true')
    parser.add_argument('--use_embodied', action='store_true')
    parser.add_argument('--use_hierarchical', action='store_true')
    parser.add_argument('--use_continual', action='store_true')
    parser.add_argument('--use_causal', action='store_true')
    
    # AGI 模块配置
    parser.add_argument('--symbolic_vocab_size', type=int, default=512)
    parser.add_argument('--n_subroutines', type=int, default=4)
    parser.add_argument('--memory_capacity', type=int, default=10000)
    parser.add_argument('--max_goal_depth', type=int, default=5)
    parser.add_argument('--n_counterfactuals', type=int, default=2)
    
    # 损失权重
    parser.add_argument('--lambda_symbolic', type=float, default=0.01)
    parser.add_argument('--lambda_self', type=float, default=0.01)
    parser.add_argument('--lambda_action', type=float, default=0.01)
    parser.add_argument('--lambda_plasticity', type=float, default=0.001)
    
    # 训练配置
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'classification'])
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--total_steps', type=int, default=10000)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--curriculum', action='store_true')
    
    # 日志与保存
    parser.add_argument('--output_dir', type=str, default='outputs/agi_cortex')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1000)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 加载 YAML 配置文件（如果提供）
    # 注意：配置文件参数会被命令行参数覆盖
    if args.config and YAML_AVAILABLE:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        # 先记录命令行显式设置的参数
        import sys
        cli_keys = set()
        for arg in sys.argv[1:]:
            if arg.startswith('--') and '=' in arg:
                cli_keys.add(arg.split('=')[0].lstrip('-'))
            elif arg.startswith('--'):
                cli_keys.add(arg.lstrip('-'))
        for key, value in config.items():
            if hasattr(args, key) and key not in cli_keys:
                setattr(args, key, value)
        print(f"[Config] 已加载配置文件: {args.config}")
    elif args.config and not YAML_AVAILABLE:
        print(f"[Warning] 请求加载配置文件 {args.config} 但 pyyaml 未安装")
    
    # 验证 d_model 可被 n_timescales 整除
    assert args.d_model % args.n_timescales == 0, \
        f"d_model ({args.d_model}) must be divisible by n_timescales ({args.n_timescales})"
    
    train_loader, val_loader = load_data(args)
    
    trainer = AGICORTEXTrainer(args)
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
