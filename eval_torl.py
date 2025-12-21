import argparse
import torch
import json
import os
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer

# 复用我们已经写好的模块
from model import TrainTicketPolicyModel
from dataset import TrainTicketRLDataset
from mock_env import TrainTicketMockEnv

class EvalConfig:
    """评估专用配置类"""
    def __init__(self, args):
        self.model_path = args.base_model_path  # 基础模型
        self.lora_path = args.checkpoint_path   # 训练好的 LoRA 权重
        self.data_path = args.data_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 评估参数
        self.max_steps_per_episode = 20
        self.temperature = 0.0  # 评估时通常使用 Greedy Search (temp=0) 以求稳定
        self.max_new_tokens = 256
        self.top_k = 1 # Greedy decoding
        
        # 必须与训练时一致的工具定义
        self.TOOLS_LIST = [
            "filesystem-list_directory", "filesystem-read_file", "filesystem-write_file",
            "rail_12306-get-current-date", "rail_12306-get-station-code-by-names",
            "rail_12306-get-tickets", 
        ]
        self.num_tools = len(self.TOOLS_LIST)
        
        # 兼容性字段 (model.py 需要读取)
        self.use_lora = True 
        self.lora_r = 16 # 这里的参数主要用于初始化结构，加载权重时会被覆盖
        self.lora_alpha = 32
        self.lora_dropout = 0.05
        self.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        self.use_amp = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Original Base Model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved checkpoint (e.g., checkpoints/final_model)")
    parser.add_argument("--data_path", type=str, default="./data/train_ticket_data.json")
    parser.add_argument("--output_file", type=str, default="eval_result.json")
    return parser.parse_args()

class TORLEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = cfg.device
        
        print(f"[Eval] Loading Base Model: {cfg.model_path}")
        print(f"[Eval] Loading LoRA Adapter: {cfg.lora_path}")
        
        # 1. 初始化模型架构
        self.policy = TrainTicketPolicyModel(cfg)
        
        # 2. 加载训练好的 LoRA 权重
        # 注意：TrainTicketPolicyModel 初始化时已经挂载了未训练的 LoRA
        # 我们需要用训练好的权重覆盖它
        from peft import PeftModel
        self.policy.model = PeftModel.from_pretrained(
            self.policy.base_model, 
            cfg.lora_path,
            is_trainable=False
        )
        self.policy.to(self.device)
        self.policy.eval() # 切换到评估模式
        
        self.tokenizer = self.policy.tokenizer
        self.env = TrainTicketMockEnv(cfg)
        
        # 加载数据
        self.dataset = TrainTicketRLDataset(cfg)
        print(f"[Eval] Dataset loaded: {len(self.dataset)} episodes")

    def parse_action(self, text):
        """解析工具调用 (与 Trainer 保持一致)"""
        try:
            if "Tool:" in text:
                segment = text.split("Tool:")[-1]
                parts = segment.split("Args:")
                tool_name = parts[0].strip()
                tool_args = parts[1].strip() if len(parts) > 1 else "{}"
                tool_args = tool_args.split("\n")[0]
                return tool_name, tool_args
            return None, None
        except:
            return None, None

    def run_eval_episode(self, episode_data):
        """运行单个测试用例"""
        prompt = episode_data['prompt']
        ground_truth = episode_data['ground_truth']
        
        self.env.reset({"ground_truth": ground_truth})
        
        full_log = f"User: {prompt}\n"
        step = 0
        done = False
        
        while not done and step < self.cfg.max_steps_per_episode:
            input_text = full_log + "Assistant:"
            inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.policy.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    tokenizer=self.tokenizer,
                    stop_strings=["Observation:", "\nUser:", "<|im_end|>"],
                    do_sample=False, # 评估时使用 Greedy Search 保证确定性
                    temperature=0.0, 
                    top_k=1
                )
            
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            full_log += f"Assistant: {generated_text}\n"
            
            # 解析与执行
            tool_name, tool_args = self.parse_action(generated_text)
            
            if tool_name:
                obs, _, done, info = self.env.step(tool_name, tool_args)
                full_log += f"Observation: {obs}\n"
            else:
                if "train-ticket-plan.json" in generated_text:
                    done = True
                else:
                    # 如果模型既没调用工具也没结束，可能是输出了一些废话，强制结束
                    if step > 10 and "Tool:" not in generated_text:
                        done = True
            
            step += 1
            
        is_success = self.env.check_success()
        return is_success, full_log

    def evaluate(self):
        print("\n>>> Starting Evaluation <<<")
        success_count = 0
        total = len(self.dataset)
        results = []
        
        # 遍历数据集
        for i in tqdm(range(total)):
            ep_data = self.dataset[i]
            success, log_text = self.run_eval_episode(ep_data)
            
            if success:
                success_count += 1
            
            results.append({
                "episode_idx": i,
                "task": ep_data.get("task_name", "unknown"),
                "success": success,
                "log": log_text
            })
            
        accuracy = success_count / total
        print(f"\n========================================")
        print(f"Final Evaluation Results:")
        print(f"Total Episodes: {total}")
        print(f"Success: {success_count}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"========================================")
        
        # 保存结果
        with open(self.cfg.output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Detailed logs saved to {self.cfg.output_file}")

if __name__ == "__main__":
    args = parse_args()
    cfg = EvalConfig(args)
    
    evaluator = TORLEvaluator(cfg)
    evaluator.evaluate()