"""
Profile parser — 解析 proton finalize 输出的 profile 文件，提取基本统计。

proton 支持三种输出格式: hatchet (默认), hatchet_msgpack, chrome_trace。
本 parser 处理 chrome_trace (JSON) 格式 — 提取 traceEvents 的 kernel 时间统计。
hatchet 格式的硬件计数器 (sm_efficiency/occupancy) 提取需 hatchet 库, 属离线分析。

用法:
    parse_chrome_trace(json_path) -> Dict
"""
import json
from typing import Dict, List


def parse_chrome_trace(json_path: str) -> Dict:
    """解析 chrome_trace JSON, 返回 kernel 时间统计。"""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[PACT ProfileParser] failed to parse {json_path}: {e}")
        return {}

    events = data.get("traceEvents", [])
    # 只保留 "X" (complete) 事件, 它们是带 dur 的区间事件 (kernel 调用)
    complete = [e for e in events if e.get("ph") == "X" and "dur" in e]
    durations_us = [e["dur"] for e in complete]  # dur 单位由 displayTimeUnit 决定

    if not durations_us:
        return {"num_kernels": 0, "total_time": 0.0, "avg_time": 0.0,
                "max_time": 0.0}

    return {
        "num_kernels": len(durations_us),
        "total_time": float(sum(durations_us)),
        "avg_time": float(sum(durations_us)) / len(durations_us),
        "max_time": float(max(durations_us)),
        "min_time": float(min(durations_us)),
    }


def parse_kernel_names(json_path: str) -> List[str]:
    """提取 chrome_trace 中的 kernel 名列表。"""
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [e.get("name", "") for e in data.get("traceEvents", [])
            if e.get("ph") == "X"]
