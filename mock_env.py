import json
import difflib
from typing import Dict, Any, Tuple, Optional

class TrainTicketMockEnv:
    """
    模拟环境：基于 Dataset 中的 Gold Trajectory 进行回放判定。
    它不再依赖全局 config，而是通过 __init__ 传入配置。
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_steps = cfg.max_steps_per_episode
        
        # 运行时状态
        self.current_episode = None
        self.step_idx = 0
        self.done = False
        
        # 当前 Episode 的标准答案数据
        self.gold_tools = []
        self.gold_args = []
        self.gold_outputs = []

    def reset(self, episode_data: Dict[str, Any]):
        """
        重置环境，加载一个新的 Episode 数据。
        Args:
            episode_data: 包含 ground_truth 的字典 (来自 dataset[i])
        """
        self.current_episode = episode_data
        self.step_idx = 0
        self.done = False
        
        # 提取标准轨迹数据 (Ground Truth)
        # 注意：dataset 返回的数据结构是 {"ground_truth": {...}}
        gt = episode_data.get('ground_truth', {})
        self.gold_tools = gt.get('tool_names', [])
        self.gold_args = gt.get('tool_args', [])
        self.gold_outputs = gt.get('output_texts', [])
        
        return "Environment Ready. Please start reasoning."

    def _normalize_args(self, args_input):
        """
        参数归一化：将 JSON 字符串或字典转换为排序后的 JSON 字符串。
        用于忽略空格和 key 顺序差异的严格比对。
        """
        try:
            if isinstance(args_input, str):
                # 尝试解析 JSON 字符串
                # 有些 args 可能是空字符串或非 JSON，需容错
                if not args_input.strip():
                    return ""
                args_dict = json.loads(args_input)
            else:
                args_dict = args_input
            
            # 重新 dump 为字符串，按 key 排序，去除多余空格
            return json.dumps(args_dict, sort_keys=True, separators=(',', ':'))
        except (json.JSONDecodeError, TypeError):
            # 如果无法解析为 JSON，则直接返回去除首尾空格的原始字符串
            return str(args_input).strip()

    def _parse_observation(self, obs_json_str):
        """
        解析数据集中的 output_texts。
        原始数据通常是 '{"type": "text", "text": "..."}' 格式。
        我们需要提取核心 "text" 字段返回给模型，避免模型被 Wrapper 格式困惑。
        """
        try:
            obs_data = json.loads(obs_json_str)
            if isinstance(obs_data, dict):
                # 优先提取 text 字段
                if 'text' in obs_data:
                    return obs_data['text']
                # 其次提取 content 字段 (有些工具可能是这个字段)
                if 'content' in obs_data:
                    return obs_data['content']
            return str(obs_data)
        except:
            # 如果不是 JSON，直接返回原始字符串
            return str(obs_json_str)

    def step(self, action_name: str, action_args: str) -> Tuple[str, float, bool, Dict]:
        """
        执行一步交互
        Returns:
            observation (str): 工具执行结果
            reward (float): 本步奖励
            done (bool): 是否结束
            info (dict): 额外信息
        """
        if self.done:
            return "Episode finished.", 0.0, True, {"status": "already_done"}

        # 1. 检查是否超出 Gold Trajectory 长度
        if self.step_idx >= len(self.gold_tools):
            self.done = True
            # 如果已经超出了标准步数还在调用工具，说明模型迷路了
            return "Error: No more steps expected in this task.", -0.5, True, {"status": "overflow"}

        # 获取当前步的“标准答案”
        expected_tool = self.gold_tools[self.step_idx]
        expected_args_raw = self.gold_args[self.step_idx]
        expected_output_raw = self.gold_outputs[self.step_idx]

        # 2. 验证工具名称 (Tool Name Matching)
        # 移除可能的空白字符
        action_name = action_name.strip()
        if action_name != expected_tool:
            # 工具选错了
            # 策略：立即报错，给予负反馈
            observation = f"Error: Expected tool '{expected_tool}', but got '{action_name}'. Check your plan."
            reward = -1.0 
            # 这里的 done=False 允许模型在同一轮次中（如果实现了 retry）或者在后续训练中修正
            # 但在简单的 GRPO rollout 中，一步错通常意味着这条轨迹废了
            return observation, reward, False, {"status": "wrong_tool", "expected": expected_tool}

        # 3. 验证参数 (Argument Matching)
        norm_pred_args = self._normalize_args(action_args)
        norm_gold_args = self._normalize_args(expected_args_raw)

        # 参数完全匹配
        if norm_pred_args == norm_gold_args:
            # HIT! 
            observation = self._parse_observation(expected_output_raw)
            reward = 0.5 # 单步正确的奖励 (不需要太高，最终奖励才是重点)
            self.step_idx += 1 
            
            # 检查是否是最后一步
            if self.step_idx >= len(self.gold_tools):
                self.done = True
            
            return observation, reward, self.done, {"status": "success"}
        
        else:
            # 工具对，但参数错
            # 这是一个关键点：我们无法真正执行错误的参数。
            # 为了辅助训练，我们返回一个通用的参数错误提示。
            observation = f"Error: Arguments mismatch for tool {action_name}. Please verify the arguments format and values."
            reward = -0.5 
            return observation, reward, False, {"status": "wrong_args"}

    def check_success(self) -> bool:
        """
        检查整个 Episode 是否完美完成。
        即：是否走完了所有标准步骤，并且没有中途退出。
        """
        return self.step_idx == len(self.gold_tools) and self.done