"""
CORTEX 通用训练脚本

支持：
- 语言建模 (Language Modeling)
- 文本分类 (Text Classification)
- 课程学习 (三阶段)
- 多目标损失
- 混合精度训练
- Checkpoint 保存/恢复
- WandB / TensorBoard 日志

Usage:
    # 语言建模
    python examples/train_cortex.py --task lm --data_path data.txt --vocab_size 32000

    # 文本分类
    python examples/train_cortex.py --task classification --data_path data.csv --num_classes 10

    # 从 checkpoint 恢复
    python examples/train_cortex.py --resume checkpoint.pt

Author: tongriyao / 田晓潼
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
from torch.cuda.amp import autocast, GradScaler

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cortex import CORTEXModel


# =============================================================================
# 数据集定义
# =============================================================================

class TokenDataset(Dataset):
    """Token 序列数据集，支持 flat token list 或 pre-tokenized sequences。"""
    def __init__(self, data, seq_len=512, is_sequences=False):
        self.seq_len = seq_len
        self.samples = []
        
        if is_sequences:
            # data: list of token sequences
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
            # data: flat list of token ids
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
    """语言建模的 collate 函数：对 input 和 label 分别做 padding。"""
    inputs, labels = zip(*batch)
    max_len = max(len(x) for x in inputs)
    
    padded_inputs = []
    padded_labels = []
    masks = []
    
    for x, y in zip(inputs, labels):
        pad_len = max_len - len(x)
        padded_inputs.append(torch.cat([x, torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        padded_labels.append(torch.cat([y, torch.full((pad_len,), -100, dtype=torch.long)]))  # -100 ignored by CE
        masks.append(torch.cat([torch.ones(len(x)), torch.zeros(pad_len)]))
    
    return torch.stack(padded_inputs), torch.stack(padded_labels), torch.stack(masks)


class TextClassificationDataset(Dataset):
    """文本分类数据集。"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.tokens = [tokenizer.encode(t, max_length=max_length, truncation=True) for t in texts]
        self.labels = labels
        self.max_length = max_length
    
    def __len__(self):
        return len(self.tokens)
    
    def __getitem__(self, idx):
        tokens = self.tokens[idx]
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        else:
            tokens = tokens[:self.max_length]
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# =============================================================================
# 简单 Tokenizer
# =============================================================================

class SimpleTokenizer:
    """简单的字符级或词级别 tokenizer（用于演示）。
    生产环境建议替换为 transformers.AutoTokenizer。
    """
    def __init__(self, vocab_size=256):
        self.vocab_size = vocab_size
        self.char_mode = vocab_size <= 512
    
    def encode(self, text, max_length=512, truncation=True):
        if self.char_mode:
            tokens = [ord(c) % self.vocab_size for c in text]
        else:
            # 简单按空格分词
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
# 训练器
# =============================================================================

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 创建输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存配置
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # 初始化日志
        self.log_file = open(self.output_dir / 'train.log', 'w')
        
        # 构建模型
        self.model = self._build_model()
        self.model.to(self.device)
        
        # 构建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.95),
        )
        
        # 学习率调度
        self.warmup_steps = args.warmup_steps
        self.total_steps = args.total_steps
        self.scheduler = None
        if args.use_scheduler:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - self.warmup_steps,
                eta_min=args.lr * 0.01,
            )
        
        # 混合精度
        self.scaler = GradScaler() if args.use_amp else None
        
        # 多目标损失权重
        self.lambda_pred = 0.0
        self.lambda_sparse = 0.0
        self.lambda_conscious = 0.0
        
        # 课程学习阶段
        self.curriculum_stage = 0
        self._update_curriculum()
        
        # 加载 checkpoint
        if args.resume:
            self._load_checkpoint(args.resume)
        
        self._log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self._log(f"Training device: {self.device}")
    
    def _build_model(self):
        args = self.args
        return CORTEXModel(
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
        )
    
    def _update_curriculum(self):
        """三阶段课程学习。"""
        stage = self.curriculum_stage
        if stage == 0:
            # 阶段一：纯连续训练
            self.lambda_pred = 0.0
            self.lambda_sparse = 0.0
            self.lambda_conscious = 0.0
            for block in self.model.layers:
                block.use_spike = False
            self._log("[Curriculum] Stage 0: Continuous-only warm-up")
        elif stage == 1:
            # 阶段二：引入脉冲
            self.lambda_pred = self.args.lambda_pred * 0.5
            self.lambda_sparse = self.args.lambda_sparse * 0.5
            self.lambda_conscious = 0.0
            for block in self.model.layers:
                block.use_spike = True
            self._log("[Curriculum] Stage 1: Introducing spikes")
        else:
            # 阶段三：完整训练
            self.lambda_pred = self.args.lambda_pred
            self.lambda_sparse = self.args.lambda_sparse
            self.lambda_conscious = self.args.lambda_conscious
            for block in self.model.layers:
                block.use_spike = True
            self._log("[Curriculum] Stage 2: Full multi-objective training")
    
    def _compute_loss(self, outputs, labels):
        """计算多目标损失。"""
        loss = outputs['loss']
        
        # 预测编码损失
        if self.lambda_pred > 0 and 'layer_info' in outputs:
            pred_loss = 0.0
            for info in outputs['layer_info']:
                if 'prediction' in info and isinstance(info['prediction'], dict):
                    p = info['prediction']
                    if 'error' in p:
                        pred_loss += p['error'].pow(2).mean()
            if isinstance(pred_loss, torch.Tensor):
                loss = loss + self.lambda_pred * pred_loss
        
        # 脉冲稀疏性损失
        if self.lambda_sparse > 0 and 'layer_info' in outputs:
            spike_rates = []
            for info in outputs['layer_info']:
                if 'spike_rate' in info:
                    spike_rates.append(info['spike_rate'])
            if spike_rates:
                mean_rate = sum(spike_rates) / len(spike_rates)
                # 鼓励脉冲率在目标值附近（默认 0.2）
                target_rate = self.args.target_spike_rate
                sparse_loss = torch.tensor((mean_rate - target_rate) ** 2, device=loss.device)
                loss = loss + self.lambda_sparse * sparse_loss
        
        # 意识状态平滑损失
        if self.lambda_conscious > 0 and 'consciousness' in outputs:
            c = outputs['consciousness']
            if c is not None and c.dim() >= 2:
                # 鼓励意识状态的变化平滑（L2正则）
                reg = c.pow(2).mean()
                loss = loss + self.lambda_conscious * reg
        
        return loss
    
    def _train_step(self, batch):
        """单次训练步。"""
        if self.args.task == 'lm':
            # collate_fn 返回 (inputs, labels, masks)
            if len(batch) == 3:
                inputs, labels, attention_mask = batch
                attention_mask = attention_mask.to(self.device)
            else:
                inputs, labels = batch
                attention_mask = (inputs != 0).float().to(self.device)
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
        else:
            inputs, labels = batch
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            attention_mask = None
        
        self.optimizer.zero_grad()
        
        # 混合精度前向
        if self.scaler:
            with autocast():
                outputs = self.model(
                    inputs,
                    labels=labels,
                    attention_mask=attention_mask,
                    return_consciousness=self.lambda_conscious > 0,
                    return_all_info=(self.lambda_pred > 0 or self.lambda_sparse > 0),
                )
                loss = self._compute_loss(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(
                inputs,
                labels=labels,
                attention_mask=attention_mask,
                return_consciousness=self.lambda_conscious > 0,
                return_all_info=(self.lambda_pred > 0 or self.lambda_sparse > 0),
            )
            loss = self._compute_loss(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            self.optimizer.step()
        
        # 学习率 warm-up
        if self.global_step < self.warmup_steps:
            lr_scale = (self.global_step + 1) / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.args.lr * lr_scale
        elif self.scheduler:
            self.scheduler.step()
        
        return loss.item(), outputs
    
    def _validate(self, val_loader):
        """验证。"""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if self.args.task == 'lm':
                    if len(batch) == 3:
                        inputs, labels, _ = batch
                    else:
                        inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                
                outputs = self.model(inputs, labels=labels)
                loss = outputs['loss']
                
                batch_size = inputs.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
        
        self.model.train()
        return total_loss / total_samples
    
    def _save_checkpoint(self, filename):
        """保存 checkpoint。"""
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args),
        }
        if self.scheduler:
            ckpt['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(ckpt, self.output_dir / filename)
    
    def _load_checkpoint(self, path):
        """加载 checkpoint。"""
        self._log(f"Resuming from {path}")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt['best_val_loss']
        if self.scheduler and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    
    def _log(self, message):
        """打印并写入日志。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        self.log_file.write(line + '\n')
        self.log_file.flush()
    
    def train(self, train_loader, val_loader=None):
        """主训练循环。"""
        self.model.train()
        step_in_epoch = 0
        epoch = 0
        
        while self.global_step < self.total_steps:
            epoch += 1
            self._log(f"Epoch {epoch} started")
            
            for batch in train_loader:
                loss, outputs = self._train_step(batch)
                self.global_step += 1
                step_in_epoch += 1
                
                # 检查课程学习阶段切换
                stage_size = self.total_steps // 3
                new_stage = min(self.global_step // stage_size, 2)
                if new_stage != self.curriculum_stage:
                    self.curriculum_stage = new_stage
                    self._update_curriculum()
                
                # 日志
                if self.global_step % self.args.log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    log_msg = f"Step {self.global_step}/{self.total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | Stage: {self.curriculum_stage}"
                    
                    # 附加脉冲率信息
                    if 'layer_info' in outputs:
                        rates = [i['spike_rate'] for i in outputs['layer_info'] if 'spike_rate' in i]
                        if rates:
                            log_msg += f" | Spike: {sum(rates)/len(rates):.3f}"
                    
                    self._log(log_msg)
                
                # 验证
                if val_loader and self.global_step % self.args.val_interval == 0:
                    val_loss = self._validate(val_loader)
                    self._log(f"Validation Loss: {val_loss:.4f}")
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint('best.pt')
                        self._log("Saved best checkpoint")
                
                # 定期保存
                if self.global_step % self.args.save_interval == 0:
                    self._save_checkpoint(f'step_{self.global_step}.pt')
                
                # 结束判断
                if self.global_step >= self.total_steps:
                    break
        
        # 保存最终模型
        self._save_checkpoint('final.pt')
        self._log("Training completed!")
        self.log_file.close()


# =============================================================================
# 数据加载
# =============================================================================

def load_data(args):
    """加载 tokenizer 和数据集。支持 HuggingFace datasets / 本地文本 / 随机数据。"""
    # ==================== Tokenizer ====================
    tokenizer = None
    if args.tokenizer_path:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            args.vocab_size = len(tokenizer)
            print(f"[Data] Loaded tokenizer: {args.tokenizer_path}, vocab_size={args.vocab_size}")
        except ImportError:
            print("[Warning] transformers not installed, falling back to SimpleTokenizer")
    
    if tokenizer is None:
        tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
        print(f"[Data] Using SimpleTokenizer, vocab_size={args.vocab_size}")
    
    # ==================== Language Modeling ====================
    if args.task == 'lm':
        token_sequences = []
        
        if args.dataset_name == 'random':
            print("[Data] Generating random data for demo")
            token_sequences = [
                torch.randint(0, args.vocab_size, (args.max_seq_len * 2,)).tolist()
                for _ in range(args.num_samples)
            ]
        
        elif args.dataset_name.endswith('.txt') or (args.data_path and args.data_path.endswith('.txt')):
            path = args.dataset_name if args.dataset_name.endswith('.txt') else args.data_path
            print(f"[Data] Loading text from {path}")
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if hasattr(tokenizer, 'encode'):
                token_ids = tokenizer.encode(text)
            else:
                token_ids = tokenizer(text)['input_ids']
            
            # 拆分为多个序列
            seq_len = args.max_seq_len * 2
            for i in range(0, len(token_ids), seq_len):
                chunk = token_ids[i:i + seq_len]
                if len(chunk) >= args.max_seq_len + 1:
                    token_sequences.append(chunk)
            print(f"[Data] Total sequences: {len(token_sequences)}")
        
        else:
            # 尝试 HuggingFace datasets
            try:
                from datasets import load_dataset
                ds_name = args.dataset_name
                ds_config = args.dataset_config
                text_col = args.text_column
                
                print(f"[Data] Loading HuggingFace dataset: {ds_name}")
                if ds_config:
                    raw_dataset = load_dataset(ds_name, ds_config)
                else:
                    raw_dataset = load_dataset(ds_name)
                
                def tokenize(examples):
                    if hasattr(tokenizer, 'encode_batch'):
                        return tokenizer(examples[text_col], truncation=True, max_length=args.max_seq_len * 2)
                    return tokenizer(examples[text_col], truncation=True, max_length=args.max_seq_len * 2)
                
                tokenized = raw_dataset.map(tokenize, batched=True, remove_columns=raw_dataset['train'].column_names)
                
                for split in ['train', 'validation']:
                    if split not in tokenized:
                        continue
                    for ex in tokenized[split]:
                        if 'input_ids' in ex:
                            token_sequences.append(ex['input_ids'])
                
                print(f"[Data] Loaded {len(token_sequences)} sequences from HF dataset")
            except ImportError:
                print("[Error] datasets library not installed. pip install datasets")
                raise
            except Exception as e:
                print(f"[Error] Failed to load dataset: {e}")
                raise
        
        # 构建数据集
        n_train = int(len(token_sequences) * 0.9)
        train_dataset = TokenDataset(token_sequences[:n_train], args.max_seq_len, is_sequences=True)
        val_dataset = TokenDataset(token_sequences[n_train:], args.max_seq_len, is_sequences=True) if n_train < len(token_sequences) else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=lambda batch: lm_collate_fn(batch, pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            drop_last=True,
            collate_fn=lambda batch: lm_collate_fn(batch, pad_token_id=tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0),
        ) if val_dataset else None
    
    # ==================== Classification ====================
    elif args.task == 'classification':
        print("[Data] Generating random classification data for demo")
        texts = [f"sample text {i}" for i in range(args.num_samples)]
        labels = [i % args.num_classes for i in range(args.num_samples)]
        dataset = TextClassificationDataset(texts, labels, tokenizer, args.max_seq_len)
        n_train = int(len(dataset) * 0.9)
        train_dataset = torch.utils.data.Subset(dataset, range(n_train))
        val_dataset = torch.utils.data.Subset(dataset, range(n_train, len(dataset)))
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    return train_loader, val_loader


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train CORTEX model')
    
    # 任务
    parser.add_argument('--task', type=str, default='lm', choices=['lm', 'classification'])
    parser.add_argument('--data_path', type=str, default=None, help='Local file path (.txt or .csv)')
    
    # 数据
    parser.add_argument('--dataset_name', type=str, default='random',
                        help='"random" | local .txt path | HuggingFace dataset name (e.g. wikitext, openwebtext)')
    parser.add_argument('--dataset_config', type=str, default=None,
                        help='HuggingFace dataset config, e.g. wikitext-2-raw-v1')
    parser.add_argument('--text_column', type=str, default='text')
    parser.add_argument('--tokenizer_path', type=str, default=None,
                        help='HuggingFace tokenizer name, e.g. gpt2, bert-base-chinese')
    parser.add_argument('--num_samples', type=int, default=1000)
    
    # 模型
    parser.add_argument('--vocab_size', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=126)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_modules', type=int, default=4)
    parser.add_argument('--workspace_dim', type=int, default=63)
    parser.add_argument('--n_branches', type=int, default=4)
    parser.add_argument('--n_timescales', type=int, default=3)
    parser.add_argument('--max_seq_len', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--tie_weights', action='store_true')
    parser.add_argument('--causal', action='store_true', default=True,
                        help='Use causal masking for autoregressive LM (default: True)')
    
    # 训练
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--total_steps', type=int, default=5000)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--use_scheduler', action='store_true', default=True)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 多目标损失
    parser.add_argument('--lambda_pred', type=float, default=0.1)
    parser.add_argument('--lambda_sparse', type=float, default=0.01)
    parser.add_argument('--lambda_conscious', type=float, default=0.01)
    parser.add_argument('--target_spike_rate', type=float, default=0.2)
    
    # （数据参数已移至上方）
    
    # 日志和保存
    parser.add_argument('--output_dir', type=str, default='outputs/cortex')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=500)
    parser.add_argument('--save_interval', type=int, default=1000)
    parser.add_argument('--resume', type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CORTEX Training")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Device: {args.device}")
    print(f"Model: d_model={args.d_model}, n_layers={args.n_layers}, causal={args.causal}")
    print(f"Training steps: {args.total_steps}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)
    
    # 加载数据
    train_loader, val_loader = load_data(args)
    print(f"Train batches: {len(train_loader)}")
    if val_loader:
        print(f"Val batches: {len(val_loader)}")
    
    # 创建训练器
    trainer = Trainer(args)
    
    # 开始训练
    trainer.train(train_loader, val_loader)


if __name__ == '__main__':
    main()
