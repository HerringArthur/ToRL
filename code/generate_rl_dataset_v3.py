# generate_rl_dataset_v2.py
"""

基于 mcp_rl_graph_v2.json（包含同名不同参的工具变体）
从原始轨迹数据重新生成 rl_dataset_llm_v2.json(用于强化学习的工具格式和调用轨迹以及返回值)


需要 running=="done" AND evaluation==true
"""

import json
import hashlib
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm


def compute_param_hash(args: Any) -> str:
    """计算参数的 schema hash（基于参数键）"""
    keys = set()
    if isinstance(args, dict):
        keys = set(args.keys())
    elif isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                keys = set(parsed.keys())
        except:
            pass
    
    if not keys:
        return ""
    return hashlib.md5(str(sorted(keys)).encode()).hexdigest()[:12]


def load_mcp_graph_v2(path: str) -> Tuple[Dict[str, Dict[str, int]], Dict[int, str], int]:
    """
    加载 mcp_rl_graph_v2.json

    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    tools = data.get("tools", [])
    
    # 构建映射: original_name -> {param_hash -> tool_id}
    tool_variant_map: Dict[str, Dict[str, int]] = defaultdict(dict)
    id_to_name: Dict[int, str] = {}
    
    for tool in tools:
        tool_id = tool.get("id")
        name = tool.get("name")
        original_name = tool.get("original_name", name)
        key_hash = tool.get("key_hash", "")
        
        if tool_id is None or name is None:
            continue
        
        id_to_name[tool_id] = name
        
        # 映射: original_name + param_hash -> tool_id
        if key_hash:
            tool_variant_map[original_name][key_hash] = tool_id
        else:
            # 没有 key_hash 的工具，用空字符串作为默认
            tool_variant_map[original_name][""] = tool_id
    
    print(f"[load_mcp_graph_v2] Loaded {len(tools)} tools")
    print(f"[load_mcp_graph_v2] Unique original names: {len(tool_variant_map)}")
    
    # 统计有多少工具有多个变体
    multi_variant = sum(1 for v in tool_variant_map.values() if len(v) > 1)
    print(f"[load_mcp_graph_v2] Tools with multiple variants: {multi_variant}")
    
    return dict(tool_variant_map), id_to_name, len(tools)


def resolve_tool_id(
    tool_name: str,
    arguments: Any,
    tool_variant_map: Dict[str, Dict[str, int]],
) -> Optional[int]:
    """
    根据工具名和参数解析 tool_id
    
    如果有多个变体，根据参数的 key_hash 匹配
    如果没有匹配的变体，返回默认变体（如果存在）
    """
    if tool_name not in tool_variant_map:
        return None
    
    variants = tool_variant_map[tool_name]
    
    # 计算参数的 key_hash
    param_hash = compute_param_hash(arguments)
    
    # 精确匹配
    if param_hash in variants:
        return variants[param_hash]
    
    # 没有精确匹配，尝试找最佳匹配
    # 1. 尝试空 hash（默认变体）
    if "" in variants:
        return variants[""]
    
    # 2. 返回第一个变体
    return next(iter(variants.values()))


def parse_arguments(args_str: str) -> Dict:
    """解析参数字符串为字典"""
    if not args_str:
        return {}
    
    try:
        parsed = json.loads(args_str)
        if isinstance(parsed, dict):
            return parsed
    except:
        pass
    
    return {}


def process_trajectory_file(
    filepath: str,
    tool_variant_map: Dict[str, Dict[str, int]],
) -> List[Dict]:
    """
    处理单个轨迹文件
    
    返回 episode 列表
    """
    episodes = []
    
    with open(filepath, 'r') as f:
        for line in f:
            try:
                traj = json.loads(line.strip())
            except:
                continue
            
            # 提取基本信息
            task_name = traj.get("task_name", "")
            
            # task_status 是 JSON 字符串，需要解析
            task_status_str = traj.get("task_status", "{}")
            try:
                task_status = json.loads(task_status_str) if isinstance(task_status_str, str) else task_status_str
                
                # 正确的成功判断：
                # running == "done" AND evaluation == true
                running_done = task_status.get("running") == "done"
                evaluation_passed = task_status.get("evaluation") == True
                success = 1 if (running_done and evaluation_passed) else 0
                
            except:
                success = 0
            
            # messages 是 JSON 字符串，需要解析
            messages_str = traj.get("messages", "[]")
            try:
                messages = json.loads(messages_str) if isinstance(messages_str, str) else messages_str
            except:
                messages = []
            
            if not isinstance(messages, list):
                continue
            
            # 提取 user_prompt（第一条 user 消息）
            user_prompt = ""
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        user_prompt = content
                    elif isinstance(content, list):
                        # content 可能是 list of {type, text}
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_prompt = item.get("text", "")
                                break
                    break
            
            # 提取工具调用序列
            tool_ids = []
            tool_names = []
            tool_args = []
            output_texts = []
            param_hashes = []
            
            # 收集所有工具调用和输出
            tool_call_map = {}  # tool_call_id -> (name, args)
            
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                
                role = msg.get("role", "")
                
                if role == "assistant":
                    tool_calls = msg.get("tool_calls", [])
                    if not isinstance(tool_calls, list):
                        continue
                    
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        
                        tc_id = tc.get("id", "")
                        func = tc.get("function", {})
                        if not isinstance(func, dict):
                            continue
                        
                        name = func.get("name", "")
                        args_str = func.get("arguments", "")
                        
                        if not name:
                            continue
                        
                        # 解析参数
                        args = parse_arguments(args_str)
                        
                        # 解析 tool_id（考虑参数变体）
                        tool_id = resolve_tool_id(name, args, tool_variant_map)
                        
                        if tool_id is None:
                            # 工具名不在图中，跳过
                            continue
                        
                        tool_ids.append(tool_id)
                        tool_names.append(name)
                        tool_args.append(args_str if args_str else "{}")
                        param_hashes.append(compute_param_hash(args))
                        
                        # 记录 tool_call_id 用于匹配输出
                        tool_call_map[tc_id] = len(tool_ids) - 1
                
                elif role == "tool":
                    # 工具输出
                    tc_id = msg.get("tool_call_id", "")
                    content = msg.get("content", "")
                    
                    if isinstance(content, list):
                        # 可能是 [{type: text, text: ...}]
                        text_parts = []
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                        content = " ".join(text_parts)
                    
                    output_texts.append(str(content)[:500])  # 截断
            
            # 确保输出数量匹配
            while len(output_texts) < len(tool_ids):
                output_texts.append("")
            output_texts = output_texts[:len(tool_ids)]
            
            if tool_ids:
                episodes.append({
                    "task_name": task_name,
                    "user_prompt": user_prompt[:2000],  # 截断
                    "success": int(success),
                    "tool_ids": tool_ids,
                    "tool_names": tool_names,
                    "tool_args": tool_args,
                    "output_texts": output_texts,
                    "param_hashes": param_hashes,
                })
    
    return episodes


def generate_rl_dataset_v2(
    trajectories_dir: str,
    mcp_graph_path: str,
    output_path: str,
):
    """
    生成 RL Dataset V2
    """
    print("=" * 70)
    print("Generating RL Dataset V2.1")
    print("Fixed: success = (running=='done' AND evaluation==true)")
    print("=" * 70)
    
    # 1. 加载工具图
    tool_variant_map, id_to_name, num_tools = load_mcp_graph_v2(mcp_graph_path)
    
    # 2. 处理轨迹文件
    trajectories_dir = Path(trajectories_dir)
    all_episodes = []
    
    jsonl_files = list(trajectories_dir.glob("*.jsonl"))
    print(f"\n[Processing] Found {len(jsonl_files)} trajectory files")
    
    for filepath in tqdm(jsonl_files, desc="Processing trajectories"):
        episodes = process_trajectory_file(str(filepath), tool_variant_map)
        all_episodes.extend(episodes)
    
    print(f"\n[Result] Generated {len(all_episodes)} episodes")
    
    # 3. 统计
    success_count = sum(1 for ep in all_episodes if ep["success"])
    total_tools = sum(len(ep["tool_ids"]) for ep in all_episodes)
    unique_tools = set()
    for ep in all_episodes:
        unique_tools.update(ep["tool_ids"])
    
    print(f"[Stats] Success episodes: {success_count}/{len(all_episodes)} ({100*success_count/len(all_episodes):.1f}%)")
    print(f"[Stats] Total tool calls: {total_tools}")
    print(f"[Stats] Unique tools used: {len(unique_tools)}")
    print(f"[Stats] Tool ID range: {min(unique_tools)} - {max(unique_tools)}")
    
    # 4. 保存
    output_data = {
        "meta": {
            "version": "v2.1",
            "num_episodes": len(all_episodes),
            "num_tools": num_tools,
            "success_episodes": success_count,
            "source": "Toolathlon-Trajectories",
            "success_criteria": "running=='done' AND evaluation==true",
            "features": [
                "tool_variants",  # 支持同名不同参的工具变体
                "param_hashes",   # 记录参数 hash
            ],
        },
        "episodes": all_episodes,
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[Saved] {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    PROJECT_ROOT = Path("/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/tool-use")
    
    trajectories_dir = PROJECT_ROOT / "Toolathlon-Trajectories"
    mcp_graph_path = PROJECT_ROOT / "json_file" / "mcp_rl_graph_v2.json"
    output_path = PROJECT_ROOT / "GRPO-ACO" / "data" / "rl_dataset_llm_v2.json"
    
    generate_rl_dataset_v2(
        str(trajectories_dir),
        str(mcp_graph_path),
        str(output_path),
    )