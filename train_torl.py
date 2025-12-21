import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path
import numpy as np

# 引入项目模块
# 确保 dataset.py, model.py, mock_env.py 在同一目录下
from model import TrainTicketPolicyModel
from dataset import TrainTicketRLDataset
from mock_env import TrainTicketMockEnv

# ============================================================================
# 1. 配置管理 (Configuration)
# ============================================================================

class RuntimeConfig:
    """
    动态配置类：将命令行参数转换为对象属性，并管理静态配置。
    """
    def __init__(self, args):
        # 从命令行参数继承
        for key, value in vars(args).items():
            setattr(self, key, value)
        
        # 路径处理
        self.project_root = Path(__file__).parent.absolute()
        self.output_dir = Path(args.output_dir)
        self.data_path = Path(args.data_path)
        
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # 静态配置
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # Qwen LoRA Target Modules
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        
        # 工具列表 (与数据集对齐)
        self.TOOLS_LIST = [
            "filesystem-list_directory", "filesystem-read_file", "filesystem-write_file",
            "rail_12306-get-current-date", "rail_12306-get-station-code-by-names",
            "rail_12306-get-tickets", 
            # 根据实际数据补充...
        ]
        self.num_tools = len(self.TOOLS_LIST)

def parse_args():
    parser = argparse.ArgumentParser(description="TORL (Tool-Integrated RL) Training Script")

    # --- 路径配置 ---
    parser.add_argument("--data_path", type=str, default="./data/train_ticket_data.json", help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./checkpoints_torl", help="Output directory")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model path")

    # --- 训练超参数 ---
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--group_size", type=int, default=8, help="Number of rollouts per prompt (GRPO group size)")
    parser.add_argument("--max_steps_per_episode", type=int, default=15, help="Max interaction steps")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N global steps")
    parser.add_argument("--batch_size", type=int, default=1, help="Prompt batch size (keep 1 for standard GRPO)")

    # --- LoRA 配置 ---
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # --- 生成参数 ---
    parser.add_argument("--temperature", type=float, default=1.0, help="Must be >0 for diverse sampling")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    
    # --- 奖励权重 ---
    parser.add_argument("--reward_success_weight", type=float, default=2.0)
    
    # --- 系统 ---
    parser.add_argument("--use_amp", action="store_true", default=True, help="Enable Automatic Mixed Precision")

    return parser.parse_args()

# ============================================================================
# 2. 训练器核心 (Trainer)
# ============================================================================

class TORLTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        
        # 初始化组件
        print(f"[Trainer] Initializing Model from {cfg.model_path}...")
        self.policy = TrainTicketPolicyModel(cfg)
        self.policy.to(self.device)
        self.tokenizer = self.policy.tokenizer
        
        print("[Trainer] Loading Dataset and Env...")
        self.dataset = TrainTicketRLDataset(cfg)
        # shuffle=True 保证训练样本的随机性
        self.dataloader = DataLoader(self.dataset, batch_size=cfg.batch_size, shuffle=True)
        self.env = TrainTicketMockEnv(cfg)
        
        self.optimizer = AdamW(self.policy.parameters(), lr=cfg.learning_rate)
        self.global_step = 0
        
        # 停止符 (用于生成中断)
        self.stop_strings = ["Observation:", "\nUser:", "<|im_end|>"]

    def parse_action(self, text):
        """
        解析模型输出。
        假设模型输出格式灵活，这里做简单的关键字匹配。
        """
        try:
            # 寻找最后一个 "Tool:"
            if "Tool:" in text:
                # 取最后一次调用的内容
                segment = text.split("Tool:")[-1]
                parts = segment.split("Args:")
                tool_name = parts[0].strip()
                tool_args = parts[1].strip() if len(parts) > 1 else "{}"
                # 去除可能的额外文本
                tool_args = tool_args.split("\n")[0]
                return tool_name, tool_args
            return None, None
        except:
            return None, None

    def run_rollout(self, prompt, ground_truth):
        """
        单次轨迹采样 (Inference Mode)。
        """
        self.env.reset({"ground_truth": ground_truth})
        
        # 构造初始对话历史
        # 格式: User: ... \nAssistant:
        full_text_log = f"User: {prompt}\n"
        
        total_reward = 0
        done = False
        step = 0
        success = False
        
        while not done and step < self.cfg.max_steps_per_episode:
            # 拼接 Prompt
            input_text = full_text_log + "Assistant:"
            
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            # 模型生成
            with torch.no_grad():
                outputs = self.policy.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    tokenizer=self.tokenizer,
                    stop_strings=self.stop_strings,
                    # GRPO 必须开启采样以获得多样性
                    do_sample=True 
                )
            
            # 提取新生成的文本
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 更新日志
            full_text_log += f"Assistant: {generated_text}\n"
            
            # 执行动作
            tool_name, tool_args = self.parse_action(generated_text)
            
            if tool_name:
                obs, reward, done, info = self.env.step(tool_name, tool_args)
                full_text_log += f"Observation: {obs}\n"
                total_reward += reward
            else:
                # 检查是否完成任务 (生成了最终文件)
                if "train-ticket-plan.json" in generated_text:
                    if self.env.check_success():
                        total_reward += self.cfg.reward_success_weight
                        success = True
                    done = True
                else:
                    # 格式错误或无意义输出
                    total_reward -= 0.5
                    done = True # 也可以选择不 done，让它重试，这里简化为结束
            
            step += 1

        return {
            "full_text": full_text_log,
            "final_reward": total_reward,
            "success": success
        }

    def compute_grpo_loss(self, trajectories):
        """
        计算 GRPO Loss。
        关键点：只对 Assistant 生成的 Action 部分计算 Loss，Mask 掉其他部分。
        """
        # 1. 计算优势 (Advantage)
        rewards = torch.tensor([t['final_reward'] for t in trajectories], device=self.device)
        # 组内标准化 (Group Standardization)
        if len(rewards) > 1:
            mean_reward = rewards.mean()
            std_reward = rewards.std() + 1e-8
            advantages = (rewards - mean_reward) / std_reward
        else:
            advantages = rewards # 如果 group_size=1，无法归一化

        total_loss = 0
        valid_trajs = 0

        # 2. 逐条轨迹计算 Loss
        for idx, traj in enumerate(trajectories):
            text = traj['full_text']
            adv = advantages[idx]
            
            # Tokenize 完整文本
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                max_length=2048
            ).to(self.device)
            
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
            
            # --- 构建 Mask (Labels) ---
            # 我们只需要计算 Assistant 回复部分的 Loss
            # 策略：将 User Prompt 和 Observation 部分的 label 设为 -100 (Ignored)
            labels = input_ids.clone()
            
            # 简单 Mask 逻辑：
            # 1. 找到所有 "Assistant:" 的位置作为起点
            # 2. 找到所有 "Observation:" 或 "User:" 的位置作为终点
            # 注意：这是基于文本的启发式 Mask，生产环境建议在 Token ID 层面做更精细的对齐
            
            # 简单全量 Mask 方法 (Baseline):
            # 将 user prompt 之前的部分 mask 掉。
            # 这里为了代码稳健性，采用简化的 Training：全序列训练，但依赖 Advantage 加权
            # 更佳实践是实现 ChatML 格式的 DataCollator，这里手动模拟一下
            
            # (简易版：不 Mask User Prompt，假设模型能学会 Copy。
            #  如果需要严格 Mask，需要在该 Loop 里 decode 每一段并查找 index)
            
            # Forward Pass
            outputs = self.policy.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels # HF model 内部计算 CE Loss
            )
            
            # 提取 Logits 手动计算 Loss 以便加权
            logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            
            # 平均 Token Loss
            policy_loss = token_losses.mean()
            
            # GRPO Loss: - Advantage * log_pi
            # 因为 policy_loss = - log_pi (NLL), 所以:
            # Loss = policy_loss * (-adv)
            # 当 Adv > 0, 我们希望 minimizing NLL (maximize log prob) -> loss 变负
            loss = policy_loss * (-adv)
            
            total_loss += loss
            valid_trajs += 1

        return total_loss / max(1, valid_trajs)

    def train(self):
        print(f"\n[Trainer] Starting GRPO Training")
        print(f"  - Epochs: {self.cfg.num_epochs}")
        print(f"  - Group Size: {self.cfg.group_size}")
        print(f"  - Learning Rate: {self.cfg.learning_rate}")
        print("========================================\n")
        
        self.policy.train()
        
        for epoch in range(self.cfg.num_epochs):
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}")
            
            for batch_idx, batch_data in enumerate(pbar):
                # 提取数据 (Batch Size = 1)
                prompt = batch_data['prompt'][0]
                # 处理 dataset 返回的 dict list 格式
                ground_truth = {
                    k: v[0] if isinstance(v, list) else v 
                    for k, v in batch_data['ground_truth'].items()
                }
                
                # --- 1. Rollout Phase (采样) ---
                trajectories = []
                success_count = 0
                avg_len = 0
                
                # 采样 G 条轨迹
                for _ in range(self.cfg.group_size):
                    traj = self.run_rollout(prompt, ground_truth)
                    trajectories.append(traj)
                    if traj['success']:
                        success_count += 1
                    avg_len += len(traj['full_text'])
                
                # --- 2. Training Phase (更新) ---
                self.optimizer.zero_grad()
                
                loss = self.compute_grpo_loss(trajectories)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                self.global_step += 1
                
                # --- 3. 记录与保存 ---
                avg_reward = sum(t['final_reward'] for t in trajectories) / len(trajectories)
                avg_len /= len(trajectories)
                
                pbar.set_postfix({
                    "Loss": f"{loss.item():.4f}", 
                    "Rw": f"{avg_reward:.2f}",
                    "Succ": f"{success_count}/{self.cfg.group_size}",
                    "Len": f"{int(avg_len)}"
                })
                
                if self.global_step % self.cfg.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
        
        # 训练结束保存最终模型
        self.save_checkpoint("final_model")

    def save_checkpoint(self, name):
        save_path = self.cfg.output_dir / name
        self.policy.save_pretrained(save_path)
        print(f"\n[Trainer] Checkpoint saved: {save_path}")

# ============================================================================
# 3. 主程序入口
# ============================================================================

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 构建运行时配置
    cfg = RuntimeConfig(args)
    
    # 启动训练
    trainer = TORLTrainer(cfg)
    trainer.train()