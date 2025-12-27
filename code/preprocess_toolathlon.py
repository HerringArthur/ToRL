# preprocess_toolathlon.py

"""

得到调用的轨迹json文件

顶层两个 key：meta 和 episodes

    meta：描述“全局字典”和各种 id 映射

        tool_name_to_id：key：工具名字（MCP 工具名）；value：整数 id（对应 mcp_rl_graph.json 里的 id）
        start_tool_id / start_output_id 用来给第一个状态填前情
        output_vocab：这是一个输出词表，用来给每种不同的工具输出一个 embedding id

    episodes：一条一条的 RL 轨迹（episode）

        episode_id：一般就是原始 request_id，唯一标识一条任务
        task_name：对应原始数据里的 task_name
        success：只要这条轨迹里调用过 claim_done / local-claim_done 之类的工具，就视为成功 1，否则 0
        tool_ids：这条 episode 中，按时间顺序调用的工具 id 列表
        output_ids：和 tool_ids 对齐的列表：第 t 个工具调用的返回文本，对应的 embedding id

tool_ids / output_ids 决定了每一步的真实动作和工具输出

start_tool_id / start_output_id 用来给第一个状态

success & Len(y) 被用来算 R(y)、A(y)，再被均匀分配到每一步当作 advantage

 """


# preprocess_toolathlon_llm.py
import json
from pathlib import Path
from typing import Dict, Any, List


DONE_TOOL_KEYWORDS = [
    "claim_done",
    "local-claim_done",
    "local-claim-done",
]


def _maybe_json(x):
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return x
    return x


def extract_user_prompt(messages: List[Dict[str, Any]]) -> str:
    """把所有 role=user 的内容拼成一个 prompt"""
    parts = []
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        content = m.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, dict):

            if "text" in content and isinstance(content["text"], str):
                text = content["text"]
            else:
                text = json.dumps(content, ensure_ascii=False)
        else:
            text = str(content)
        if text:
            parts.append(text.strip())
    return "\n\n".join(parts).strip()


def extract_steps_from_messages(messages: List[Dict[str, Any]]):

    steps = []

    # tool_call_id -> 工具输出
    id_to_output = {}
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "tool":
            continue
        tc_id = m.get("tool_call_id")
        if not tc_id:
            continue
        content = m.get("content")
        if isinstance(content, dict):
            out_text = content.get("text")
            if out_text is None:
                out_text = json.dumps(content, ensure_ascii=False)
        else:
            out_text = str(content)
        id_to_output[tc_id] = out_text or ""

    # assistant.tool_calls -> tool_name + 参数 + 对应输出
    for m in messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls")
        if not tcs:
            continue
        tcs = _maybe_json(tcs)
        if isinstance(tcs, dict):
            tcs = [tcs]
        if not isinstance(tcs, list):
            continue
        for tc in tcs:
            if not isinstance(tc, dict):
                continue
            func = tc.get("function") or {}
            tool_name = func.get("name")
            tc_id = tc.get("id")
            # Tool 参数就是 function.arguments（字符串）
            tool_args = func.get("arguments", "")
            if not tool_name or not tc_id:
                continue
            out_text = id_to_output.get(tc_id, "")
            steps.append(
                {
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "output_text": out_text,
                }
            )
    return steps


def parse_record(obj: Dict[str, Any]):
    """
    把一条完整的任务记录解析成 RL episode 所需信息。

    """

    raw_config = obj.get("config", None)
    config = _maybe_json(raw_config)

    if isinstance(config, dict):
        task_name = config.get("task_name", obj.get("task_name", "unknown_task"))
    else:

        task_name = obj.get("task_name", "unknown_task")

    req_id = obj.get("request_id")

    messages = _maybe_json(obj.get("messages"))
    if not isinstance(messages, list):
        return None

    user_prompt = extract_user_prompt(messages)
    steps = extract_steps_from_messages(messages)
    if not steps:
        return None

    raw_task_status = obj.get("task_status", {})
    # 这里 task_status 通常是一个 JSON 字符串，比如:
    # "{\"preprocess\": \"done\", \"running\": \"done\", \"evaluation\": false}"
    task_status = _maybe_json(raw_task_status)

    running_status = ""

    if isinstance(task_status, dict):
        # 标准情况：已经成功解析成 dict
        # 例如 {"preprocess": "done", "running": "max_turn_exceeded", "evaluation": false}
        running_status = str(task_status.get("running", "")).strip()
    else:
        # 任何非 dict 的情况一律当作失败，不做字符串模糊匹配了
        running_status = ""

    success = 1 if running_status == "done" else 0

    return {
        "task_name": task_name,
        "request_id": req_id,
        "user_prompt": user_prompt,
        "success": success,
        "steps": steps,
    }



def load_mcp_tools(mcp_graph_path: Path):
    """
    从 mcp_rl_graph.json 读取工具名 -> id 映射（608 个节点）
    """
    with mcp_graph_path.open("r", encoding="utf-8") as f:
        g = json.load(f)
    tools = g["tools"]
    name_to_id = {t["name"]: int(t["id"]) for t in tools}
    return name_to_id


def preprocess_llm(mcp_graph_path: str, traj_dir: str, output_path: str):
    """
    从 Toolathlon-Trajectories-merge 目录下所有 *.jsonl 文件中
    读取轨迹，生成 rl_dataset_llm.json。

    每个 jsonl 文件中的每一行是一个和 test_optimized.json[0] 格式一致的对象。
    """
    mcp_graph_path = Path(mcp_graph_path)
    traj_dir = Path(traj_dir)
    output_path = Path(output_path)

    tool_name_to_id = load_mcp_tools(mcp_graph_path)
    print("工具数:", len(tool_name_to_id))

    episodes = []
    file_count = 0
    rec_count = 0

    for path in sorted(traj_dir.glob("*.jsonl")):
        file_count += 1
        print("处理文件:", path)
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec = parse_record(obj)
                if rec is None:
                    continue

                steps = rec["steps"]
                tool_ids = []
                output_texts = []
                tool_args = []
                for s in steps:
                    name = s["tool_name"]
                    if name not in tool_name_to_id:
                        # 有些工具可能不在 608 节点图中，直接跳过
                        continue
                    tid = tool_name_to_id[name]
                    tool_ids.append(tid)
                    output_texts.append(s["output_text"] or "")
                    tool_args.append(s.get("tool_args", "") or "")

                if not tool_ids:
                    continue

                ep = {
                    "episode_id": rec["request_id"] or f"{rec_count}",
                    "task_name": rec["task_name"],
                    "user_prompt": rec["user_prompt"],
                    "success": int(rec["success"]),
                    "tool_ids": tool_ids,
                    "output_texts": output_texts,
                    "tool_args": tool_args,
                }
                episodes.append(ep)
                rec_count += 1

    meta = {
        "tool_name_to_id": tool_name_to_id,
        "start_tool_id": len(tool_name_to_id),
    }

    out = {"meta": meta, "episodes": episodes}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("共处理 jsonl 文件:", file_count)
    print("得到 episode 数:", len(episodes))
    print("写出到:", output_path)


if __name__ == "__main__":
    PROJECT_ROOT = Path("/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/tool-use")
    base = PROJECT_ROOT / "json_file"
    preprocess_llm(
        mcp_graph_path=str(base / "mcp_rl_graph.json"),
        traj_dir=str(PROJECT_ROOT / "Toolathlon-Trajectories-merge"),
        output_path=str("/seu_share2/home/fenglei/230250004/Agent_Tool/tool-use/tool-use/GRPO-ACO/data/rl_dataset_llm.json"),
    )
