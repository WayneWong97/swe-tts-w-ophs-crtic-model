import argparse
import json
import random
import textwrap
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


RunRecord = Tuple[str, str, Dict]


def iter_runs(jsonl_path: str) -> Iterator[RunRecord]:
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
        for line in jsonl_file:
            if not line.strip():
                continue
            entry = json.loads(line)
            instance_id = entry.get("instance_id", "")
            for key, value in entry.items():
                if key.startswith("run_") and isinstance(value, dict):
                    yield instance_id, key, value


def render_snippet(content: str, width: int = 160) -> str:
    return textwrap.shorten(content.replace("\n", " "), width=width, placeholder="...")


def main() -> None:
    parser = argparse.ArgumentParser(description="随机抽样展示 funccall-off 转换后的轨迹")
    parser.add_argument("--jsonl_path", required=True, type=str, help="转换后的 JSONL 文件")
    parser.add_argument(
        "--preview_messages", type=int, default=20, help="展示的消息数(从头开始)"
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子，便于复现采样结果")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    runs = list(iter_runs(args.jsonl_path))

    if not runs:
        print("未在文件中找到任何 run_XX 轨迹。")
        return

    instance_id, run_key, run_value = rng.choice(runs)
    print("=" * 80)
    resolved = run_value.get("resolved")
    print(f"随机样本: instance={instance_id}, run={run_key}, resolved={resolved}")

    messages = run_value.get("funccalloff_messages")
    if not messages:
        error = run_value.get("funccalloff_conversion_error")
        print(f"  无转换结果 (error={error})")
        return

    total_messages = len(messages)
    print(f"  消息总数: {total_messages}")
    preview_limit = min(args.preview_messages, total_messages)
    for i in range(preview_limit):
        message = messages[i]
        role = message.get("role")
        content = message.get("content", "")
        snippet = render_snippet(content if isinstance(content, str) else str(content))
        print(f"    [{i}] role={role}")
        print(f"         {snippet}")
    if total_messages > preview_limit:
        print(f"    ... (剩余 {total_messages - preview_limit} 条消息未展示)")


if __name__ == "__main__":
    main()
