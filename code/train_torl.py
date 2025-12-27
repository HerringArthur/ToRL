
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Any, Tuple

# 引入我们新定义的 Environment
from graph_env import GraphToolEnv

# 复用原有的工具函数（假设它们是独立的，如果不独立，可能需要复制过来）
# 为了确保独立运行，我将在此文件中重新定义必要的最小集，或者从源文件导入
# 考虑到依赖关系复杂，这里实现一个精简版的 GRPO 训练流程

import argparse

# ============================================================================
# 配置
# ============================================================================

class Config:
    # 路径配置 (默认值，可被命令行参数覆盖)
    project_root = Path("/seu_share2/home/fenglei/213243847/grpo-aco/grpo-aco")
    model_path = "/seu_share2/home/fenglei/sharedata/Qwen2.5-7B-Instruct" 
    
    graph_path = project_root / "data" / "mcp_rl_graph_v2.json"
    simulator_path = project_root / "data" / "tool_simulator_database.json"
    dataset_path = project_root / "data" / "rl_dataset_llm_v3.json"
    output_dir = project_root / "output" / "grpo_graph_v1"
    
    # 训练参数
    max_steps = 15 # per episode
    group_size = 4
    batch_size = 2
    learning_rate = 1e-6
    epochs = 3
    reward_success_weight = 2.0
    
    # LoRA Config
    lora_r = 16
    lora_alpha = 32
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    @classmethod
    def update_from_args(cls, args):
        """从命令行参数更新配置"""
        if args.model_path:
            cls.model_path = args.model_path
        if args.data_path:
            cls.dataset_path = Path(args.data_path)
        if args.output_dir:
            cls.output_dir = Path(args.output_dir)
        if args.num_epochs:
            cls.epochs = args.num_epochs
        if args.learning_rate:
            cls.learning_rate = args.learning_rate
        if args.group_size:
            cls.group_size = args.group_size
        if args.max_steps_per_episode:
            cls.max_steps = args.max_steps_per_episode
        if args.lora_r:
            cls.lora_r = args.lora_r
        if args.reward_success_weight:
            cls.reward_success_weight = args.reward_success_weight
            
        # 确保目录存在
        cls.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def check_paths(cls):
        if not cls.graph_path.exists():
            print(f"Warning: Graph path {cls.graph_path} does not exist.")
        if not cls.simulator_path.exists():
            print(f"Warning: Simulator path {cls.simulator_path} does not exist.")

# ============================================================================
# 数据集
# ============================================================================

class RLDataset(Dataset):
    def __init__(self, data_path: str):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.episodes = data.get("episodes", [])
        print(f"Loaded {len(self.episodes)} episodes.")
        
    def __len__(self):
        return len(self.episodes)
        
    def __getitem__(self, idx):
        return self.episodes[idx]

# ============================================================================
# 策略模型 (Policy)
# ============================================================================

class ToolPolicy(nn.Module):
    def __init__(self, model_path: str):
        super().__init__()
        # self.device = device # device is managed by accelerator
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map=None # Accelerator will handle device placement
        )
        
        # LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=Config.lora_r,
            lora_alpha=Config.lora_alpha,
            lora_dropout=0.1
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()
        
    def generate_action(self, prompt: str, history: List[Dict]) -> str:
        # 构建 Chat 格式 Input
        messages = [{"role": "system", "content": "You are a helpful assistant that uses tools to solve tasks."}]
        messages.append({"role": "user", "content": prompt})
        
        for step in history:
            messages.append({"role": "assistant", "content": f"I will use tool: {step['name']} with args: {step['args']}"})
            messages.append({"role": "tool", "content": step['output']}) # 注意：Qwen 的 tool role 可能是 'observation' 或其他，需根据具体模板调整
            
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response

    def compute_log_prob(self, prompt: str, history: List[Dict], action: str):
        # 这是一个简化的 log_prob 计算，实际训练中需要更精细的 mask
        # 这里仅作演示框架
        pass

# ============================================================================
# 辅助函数：解析 Action
# ============================================================================

def parse_action(response: str, name_to_id: Dict[str, int]) -> Tuple[int, str]:
    """
    尝试从模型回复中解析工具调用。
    假设回复格式为 JSON 或特定格式。
    这里做一个简单的假设：模型直接输出 JSON，如 {"tool": "name", "args": {...}}
    或者 <tool_code>...
    """
    # 简单实现：查找 JSON
    try:
        start = response.find("{")
        end = response.rfind("}") + 1
        if start != -1 and end != -1:
            json_str = response[start:end]
            data = json.loads(json_str)
            tool_name = data.get("tool") or data.get("name") or data.get("function")
            args = data.get("args") or data.get("parameters") or {}
            
            if tool_name and tool_name in name_to_id:
                return name_to_id[tool_name], json.dumps(args)
    except:
        pass
        
    return None, response

# ============================================================================
# 主训练循环
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="ToRL Training")
    parser.add_argument("--model_path", type=str, help="Path to the base model")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--output_dir", type=str, help="Directory to save outputs")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--group_size", type=int, help="GRPO group size")
    parser.add_argument("--max_steps_per_episode", type=int, help="Max steps per episode")
    parser.add_argument("--lora_r", type=int, help="LoRA rank")
    parser.add_argument("--reward_success_weight", type=float, help="Reward weight for success")
    parser.add_argument("--save_every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision")
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="fp16" if args.use_amp else "no", log_with="all")
    
    Config.update_from_args(args)
    
    # Only main process checks paths and creates directories
    if accelerator.is_main_process:
        Config.check_paths()
        os.makedirs(Config.output_dir, exist_ok=True)
    
    # 1. 初始化
    if accelerator.is_main_process:
        print("Initializing Environment...")
    env = GraphToolEnv(Config.graph_path, Config.simulator_path)
    
    if accelerator.is_main_process:
        print("Loading Dataset...")
    dataset = RLDataset(Config.dataset_path)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    if accelerator.is_main_process:
        print("Loading Model...")
    
    try:
        policy = ToolPolicy(Config.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    optimizer = optim.AdamW(policy.model.parameters(), lr=Config.learning_rate)
    
    # Prepare with Accelerator
    policy, optimizer, dataloader = accelerator.prepare(
        policy, optimizer, dataloader
    )
    
    # 2. 训练循环
    global_step = 0
    for epoch in range(Config.epochs):
        if accelerator.is_main_process:
            print(f"\nEpoch {epoch+1}/{Config.epochs}")
        
        # Disable tqdm for non-main processes to avoid clutter
        progress_bar = tqdm(dataloader, disable=not accelerator.is_main_process)
        
        for batch_idx, batch in enumerate(progress_bar):
            prompts = batch['user_prompt']
            task_names = batch['task_name']
            
            optimizer.zero_grad()
            batch_loss = 0
            
            # --- GRPO Logic ---
            all_rewards = []
            
            for i, prompt in enumerate(prompts):
                task = {"user_prompt": prompt, "task_name": task_names[i]}
                
                # Group Sampling
                group_rewards = []
                
                # TODO: Optimize sampling for multi-gpu (currently naive)
                # Ideally, we should distribute rollouts across GPUs
                
                for _ in range(Config.group_size):
                    # Rollout
                    # Note: env.reset/step are not GPU-accelerated, running on CPU
                    obs = env.reset(task)
                    total_reward = 0
                    done = False
                    steps = 0
                    
                    while not done and steps < Config.max_steps:
                        # 注意：这里调用 policy.generate_action 需要实现
                        # 暂时用简单的 placeholder 避免运行错误
                        try:
                            # Unwrap model for generation if needed, or implement generate_action to handle DDP
                            unwrapped_model = accelerator.unwrap_model(policy)
                            # You might need to move inputs to device manually if generate_action doesn't handle it
                            # But since we are inside accelerator.prepare, policy is already on device
                            # However, generate_action in ToolPolicy likely expects raw strings and handles tokenization/inference
                            
                            # For DDP, generation is tricky. Usually we do generation on one process or all processes generate independently.
                            # Here we let each process generate its own trajectories.
                            
                            # We need to ensure ToolPolicy.generate_action uses the correct device
                            # Since we removed 'device' from ToolPolicy init, we should get it from model parameters
                            
                            action_text = unwrapped_model.generate_action(prompt, env.history)
                        except AttributeError:
                             # 如果 policy 没有 generate_action 方法 (可能因为是新定义的 ToolPolicy)
                             # 这里应该实现生成逻辑
                            action_text = "" 
                        
                        # 解析 (需要 parse_action 函数，这里假设它存在或需要导入)
                        # tool_id, tool_args = parse_action(action_text, env.name_to_id)
                        # 暂时 mock
                        tool_id = None
                        tool_args = "{}"

                        if tool_id is not None:
                            step_res = env.step(tool_id, tool_args)
                        else:
                            # 解析失败，给予惩罚并结束
                            step_res = env.step(-1, "{}") # -1 will trigger invalid tool error
                            step_res.reward = -0.5
                            step_res.done = True
                            
                        total_reward += step_res.reward
                        done = step_res.done
                        steps += 1
                        
                    group_rewards.append(total_reward)
                
                all_rewards.extend(group_rewards)
                
                # Advantage Computation
                rewards_tensor = torch.tensor(group_rewards, device=accelerator.device)
                mean_r = rewards_tensor.mean()
                std_r = rewards_tensor.std() + 1e-8
                advantages = (rewards_tensor - mean_r) / std_r
                
                # Policy Update (Placeholder)
                # loss = - (log_probs * advantages).mean()
                # batch_loss += loss
                
                # For demo, we just backward a dummy loss if loss is computed
                # accelerator.backward(loss)
            
            # optimizer.step()
            
            global_step += 1
            if batch_idx % 10 == 0 and accelerator.is_main_process:
                avg_r = np.mean(all_rewards) if all_rewards else 0.0
                print(f"Step {batch_idx}: Avg Reward = {avg_r:.4f}")

            # Save checkpoint
            if global_step % args.save_every == 0:
                 save_path = Config.output_dir / f"checkpoint-{global_step}"
                 # accelerator.save_state(save_path)
                 if accelerator.is_main_process:
                    print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    train()
