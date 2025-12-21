import json
import torch
from torch.utils.data import Dataset

class ToolMapper:
    """
    工具名称与 ID 的映射辅助类。
    现在它不再依赖全局 cfg，而是需要传入 tools_list 进行初始化。
    """
    def __init__(self, tools_list):
        self.tools_list = tools_list
        self.name2id = {name: idx for idx, name in enumerate(tools_list)}
        self.id2name = {idx: name for idx, name in enumerate(tools_list)}

    def get_id(self, name):
        return self.name2id.get(name)

    def get_name(self, idx):
        return self.id2name.get(idx, "unknown_tool")

    def __len__(self):
        return len(self.tools_list)


def load_raw_episodes(data_path, tools_list=None):
    """
    加载原始 JSON 数据并进行基础清洗。
    Args:
        data_path: 数据集路径
        tools_list: 可选，用于过滤包含未知工具的数据
    """
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"[Dataset] Error: Data file not found at {data_path}")
        return []
    
    # 兼容两种格式：直接列表 或 字典中包含 "episodes"
    raw_episodes = data['episodes'] if isinstance(data, dict) and 'episodes' in data else data
    
    valid_episodes = []
    skipped = 0
    
    # 允许的工具集合
    allowed_tools = set(tools_list) if tools_list else None
    
    for ep in raw_episodes:
        # 基础校验：必须包含 prompt 和工具调用序列
        if 'user_prompt' not in ep or 'tool_names' not in ep:
            skipped += 1
            continue
            
        # 校验工具名称是否都在我们的列表中
        if allowed_tools:
            tool_names = ep['tool_names']
            if not all(name in allowed_tools for name in tool_names):
                skipped += 1
                continue
            
        valid_episodes.append(ep)
        
    print(f"[Dataset] Loaded {len(valid_episodes)} valid episodes from {data_path} (Skipped {skipped})")
    return valid_episodes


class TrainTicketRLDataset(Dataset):
    """
    用于 RL (GRPO) 阶段的数据集。
    """
    def __init__(self, cfg, episodes=None):
        self.cfg = cfg
        # 在内部初始化 Mapper，如果需要用到 ID 转换
        self.tool_mapper = ToolMapper(cfg.TOOLS_LIST)
        
        if episodes is not None:
            self.episodes = episodes
        else:
            # 显式传入路径和工具列表
            self.episodes = load_raw_episodes(cfg.data_path, cfg.TOOLS_LIST)

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        # RL 阶段主要需要 Prompt 来启动生成，以及 Gold 数据来做奖励计算 (MockEnv)
        ep = self.episodes[idx]
        return {
            "task_name": ep.get("task_name", ""),
            "prompt": ep["user_prompt"],
            "ground_truth": {
                "tool_names": ep["tool_names"],
                "tool_args": ep["tool_args"],
                "output_texts": ep["output_texts"],
                # 用于 MockEnv 匹配参数
                "param_hashes": ep.get("param_hashes", []) 
            }
        }


class TrainTicketSLDataset(Dataset):
    """
    用于 SL (Supervised Learning) 预训练阶段的数据集 (Warmup)。
    """
    def __init__(self, cfg, max_history=5):
        self.cfg = cfg
        self.max_history = max_history
        self.tool_mapper = ToolMapper(cfg.TOOLS_LIST)
        
        episodes = load_raw_episodes(cfg.data_path, cfg.TOOLS_LIST)
        self.samples = []
        self._build_samples(episodes)

    def _build_state_text(self, prompt, history):
        """
        构建输入给模型的 Prompt 文本。
        """
        # 简单模板示例
        lines = [f"User: {prompt}\nHistory:"]
        if not history:
            lines.append("  (None)")
        else:
            for h in history:
                # 截断输出防止过长
                output_preview = h['output'][:100].replace('\n', ' ') 
                lines.append(f"  - Action: {h['tool']}\n  - Result: {output_preview}")
        lines.append("\nAssistant: Next Tool:")
        return "\n".join(lines)

    def _build_samples(self, episodes):
        for ep in episodes:
            prompt = ep['user_prompt']
            tool_names = ep['tool_names']
            output_texts = ep['output_texts']
            
            # 将每一步拆解为一个训练样本
            for t in range(len(tool_names)):
                # 1. 构建历史 Context
                history = []
                # 取最近 N 步历史
                start_hist = max(0, t - self.max_history)
                for i in range(start_hist, t):
                    history.append({
                        "tool": tool_names[i],
                        "output": output_texts[i]
                    })
                
                state_text = self._build_state_text(prompt, history)
                
                # 2. 获取当前步的 Label (Action ID)
                current_tool_name = tool_names[t]
                action_id = self.tool_mapper.get_id(current_tool_name)
                
                self.samples.append({
                    "state_text": state_text,
                    "action_id": action_id,
                    "task_name": ep.get("task_name", "")
                })
        
        print(f"[SL Dataset] Generated {len(self.samples)} training samples (state-action pairs)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collate_sl_batch(batch):
    """
    SL 训练的 Batch 处理函数
    """
    state_texts = [item['state_text'] for item in batch]
    action_ids = torch.tensor([item['action_id'] for item in batch], dtype=torch.long)
    return state_texts, action_ids