
import json
import random
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional, Set
from dataclasses import dataclass
from llm_tool_simulator2 import LLMToolSimulator, TaskCompletionChecker, TaskCompletionResult

@dataclass
class EnvStepResult:
    observation: str
    reward: float
    done: bool
    info: Dict[str, Any]

class GraphToolEnv:
    """
    基于工具图和模拟器的强化学习环境。
    
    State: 
        - current_tool_id: 当前所在的工具节点 ID
        - history: 历史交互记录 (List[Dict])
    
    Action:
        - tool_id: 下一步要调用的工具 ID
        - arguments: 工具参数
        
    Transition:
        - 只能跳转到 current_tool_id 的 next_tools 中定义的工具。
        - 如果跳转非法，环境结束或给予惩罚。
        
    Reward:
        - 稀疏奖励：只有在任务完成时给予正奖励。
        - 步骤惩罚：每一步给予微小负奖励以鼓励短路径。
    """
    
    def __init__(
        self, 
        graph_path: str, 
        simulator_path: str,
        device: str = "cpu"
    ):
        self.graph_path = graph_path
        self.simulator_path = simulator_path
        
        # 1. 加载工具图
        self._load_graph(graph_path)
        
        # 2. 加载模拟器 (使用 LLMToolSimulator)
        self.simulator = LLMToolSimulator(tool_system=self)
        
        # 3. 任务完成检测器
        self.checker = TaskCompletionChecker()
        
        # 状态变量
        self.current_tool_id: Optional[int] = None
        self.history: List[Dict] = []
        self.current_task: Dict = {}
        self.steps = 0
        self.max_steps = 15
        
    def _load_graph(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        self.tools_list = data.get("tools", [])
        self.tools_map = {} # id -> tool_info
        self.name_to_id = {}
        
        # 为 LLMToolSimulator 准备的数据结构
        self.extended_tools = {}
        self.original_tools = {}
        
        for tool in self.tools_list:
            tid = tool.get("id")
            self.tools_map[tid] = tool
            self.name_to_id[tool.get("name")] = tid
            
            # 填充 tool_system 接口数据
            variant_name = tool.get("name")
            original_name = tool.get("original_name", variant_name)
            
            self.extended_tools[variant_name] = tool
            # 注意：如果有多个变体对应同一个 original_name，这里会覆盖，但通常描述和参数是一样的
            if original_name not in self.original_tools:
                self.original_tools[original_name] = tool
            
        # 构建 NetworkX 图用于分析（可选）
        self.graph = nx.DiGraph()
        for tool in self.tools_list:
            tid = tool.get("id")
            self.graph.add_node(tid, name=tool.get("name"))
            for next_name, weight in tool.get("next_tools", {}).items():
                next_id = self.name_to_id.get(next_name)
                if next_id is not None:
                    self.graph.add_edge(tid, next_id, weight=weight)
                    
        print(f"[GraphToolEnv] Loaded {len(self.tools_list)} tools, {self.graph.number_of_edges()} edges.")

    def original_name_from_extended(self, variant_name: str) -> str:
        """获取工具变体的原始名称（ToolSimulator 接口）"""
        tool = self.extended_tools.get(variant_name)
        if tool:
            return tool.get("original_name", variant_name)
        return variant_name

    def reset(self, task: Dict) -> str:
        """
        重置环境到初始状态。
        
        Args:
            task: 包含 'user_prompt', 'task_name' 等信息的字典
            
        Returns:
            initial_observation: 用户输入的 Prompt
        """
        self.current_task = task
        self.history = []
        self.steps = 0
        
        # 初始状态：通常没有选中任何工具，或者处于一个虚拟的 "Start" 节点
        # 这里我们假设初始状态是空的，Agent 第一步可以选择任何 入度为0 的节点，或者所有节点？
        # 根据 mcp_rl_graph_v2.json 的逻辑，通常是由 User Prompt 触发第一个工具。
        # 为了简化，我们允许第一步选择任何工具，或者根据 Dataset 中的第一个工具作为提示（如果是 Imitation Learning）。
        # 在 RL 设定中，Policy 根据 User Prompt 生成第一个工具。
        
        self.current_tool_id = None 
        
        return task.get("user_prompt", "")

    def get_valid_next_tools(self) -> Set[int]:
        """获取当前状态下合法的后续工具 ID 集合"""
        if self.current_tool_id is None:
            # 初始状态，允许所有工具（或者基于某种检索策略，这里暂定所有）
            return set(self.tools_map.keys())
        
        current_tool_info = self.tools_map.get(self.current_tool_id)
        if not current_tool_info:
            return set()
            
        next_tools = current_tool_info.get("next_tools", {})
        valid_ids = set()
        for name in next_tools.keys():
            nid = self.name_to_id.get(name)
            if nid is not None:
                valid_ids.add(nid)
        return valid_ids

    def step(self, tool_id: int, tool_args: str) -> EnvStepResult:
        """
        执行一步交互
        
        Args:
            tool_id: 选定的工具 ID
            tool_args: 工具参数（JSON 字符串）
        """
        self.steps += 1
        
        # 1. 检查合法性
        valid_next = self.get_valid_next_tools()
        
        # 如果不是第一步，且选了非法的后续节点
        if self.current_tool_id is not None and tool_id not in valid_next:
            return EnvStepResult(
                observation="Error: Invalid tool transition.",
                reward=-1.0, # 惩罚非法跳转
                done=True,
                info={"error": "invalid_transition", "valid_tools": list(valid_next)}
            )
            
        # 2. 执行工具（模拟）
        tool_info = self.tools_map.get(tool_id)
        if not tool_info:
            return EnvStepResult(
                observation="Error: Tool ID not found.",
                reward=-0.5,
                done=True,
                info={"error": "unknown_tool"}
            )
            
        tool_name = tool_info.get("name")
        
        # 使用模拟器获取输出
        output = self.simulator.get_output(tool_name, tool_args)
        
        # 3. 更新历史
        self.current_tool_id = tool_id
        step_record = {
            "name": tool_name,
            "args": tool_args,
            "output": output
        }
        self.history.append(step_record)
        
        # 4. 检查任务完成情况 (Outcome Feedback)
        # 优先使用 GT 匹配（如果 task 中有 ground truth），其次使用 Checker
        
        done = False
        reward = 0.0
        info = {}
        
        # 简单步骤惩罚
        reward -= 0.05
        
        # 检查是否达到最大步数
        if self.steps >= self.max_steps:
            done = True
            info["reason"] = "max_steps"
            
        # 调用 Checker 进行评估
        # 使用 check_completion 方法（支持 smart_mode: 先启发式，必要时 LLM）
        check_result = self.checker.check_completion(
            self.current_task.get("task_name", ""),
            self.current_task.get("user_prompt", ""),
            self.history,
            use_cache=True,
            smart_mode=True
        )
        
        if check_result.is_complete:
            # 这里的 reward 可以是 check_result.confidence * check_result.quality_score
            reward += 1.0 * check_result.quality_score
            done = True
            info["success"] = True
            info["reason"] = check_result.reason
        elif done: # Max steps reached and not complete
            reward -= 1.0 # 任务失败惩罚
            info["success"] = False
            
        return EnvStepResult(
            observation=output,
            reward=reward,
            done=done,
            info=info
        )
