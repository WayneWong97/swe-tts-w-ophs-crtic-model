import os
import json
import gc
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import torch
from transformers import AutoTokenizer
from tqdm import tqdm
from vllm import LLM

# --- 配置 ---
MODEL_PATH = "/scratch/ywxzml3j/user72/models/openhands-critic-32b-exp-20250417"
INPUT_FILE = "tts_funccalloff_all.json"
OUTPUT_FILE_TEMPLATE = "scored_results_scheme_8_{scheme}.jsonl"  # 新文件名

# ============================ 关键参数 ============================
# 这个值现在是 VLLM 的硬限制，也是我们手动截断的目标长度
REALISTIC_MAX_LENGTH = 30720
# =================================================================

# --- 1. 加载 vllm 引擎和 Tokenizer ---

print("Loading VLLM engine...")
llm_engine = LLM(
    model=MODEL_PATH,
    task="token_reward",
    trust_remote_code=True,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len=REALISTIC_MAX_LENGTH,
    gpu_memory_utilization=0.95
)
print("VLLM engine loaded successfully.")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.truncation_side = 'left'
print(f"Tokenizer configured with truncation_side='{tokenizer.truncation_side}'")


# --- 2. 打分函数 ---

@dataclass
class FormattedSegment:
    role: str
    text: str
    start: int
    end: int
    token_indices: Optional[List[int]] = None


def build_segments_from_messages(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer,
                                 max_tokens: int) -> tuple[str, List[FormattedSegment]]:
    if not messages:
        return "", []

    raw_segments = []
    total_tokens = 0
    for msg in messages:
        role = msg.get('role', 'assistant')
        content = msg.get('content', '') or ""
        formatted = f"{role.upper()}:\n{content.strip()}\n\n"
        token_ids = tokenizer.encode(formatted, add_special_tokens=False)
        token_count = len(token_ids)
        total_tokens += token_count
        raw_segments.append({'role': role, 'text': formatted, 'token_count': token_count})

    start_idx = 0
    while total_tokens > max_tokens and start_idx < len(raw_segments):
        total_tokens -= raw_segments[start_idx]['token_count']
        start_idx += 1

    trimmed = raw_segments[start_idx:]
    text_parts = []
    segments: List[FormattedSegment] = []
    cursor = 0
    for item in trimmed:
        text_parts.append(item['text'])
        start = cursor
        cursor += len(item['text'])
        segments.append(FormattedSegment(role=item['role'], text=item['text'], start=start, end=cursor))

    final_text = ''.join(text_parts)
    return final_text, segments


def align_tokens_to_segments(text: str, segments: List[FormattedSegment], tokenizer: AutoTokenizer) -> tuple[list[int], list[tuple[int, int]]]:
    if not text or not segments:
        return [], []
    encoding = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    offsets = encoding.get("offset_mapping") or []
    input_ids = encoding["input_ids"]

    seg_idx = 0
    for token_idx, (start, end) in enumerate(offsets):
        if seg_idx >= len(segments):
            break
        if end <= start:
            continue
        mid = (start + end) / 2
        while seg_idx < len(segments) and mid >= segments[seg_idx].end:
            seg_idx += 1
        if seg_idx == len(segments):
            break
        if segments[seg_idx].start <= mid < segments[seg_idx].end:
            if segments[seg_idx].token_indices is None:
                segments[seg_idx].token_indices = []
            segments[seg_idx].token_indices.append(token_idx)
    return input_ids, offsets


def _prepare_text_and_segments(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> tuple[str, List[FormattedSegment]]:
    text, segments = build_segments_from_messages(messages, tokenizer, REALISTIC_MAX_LENGTH)
    if not text or not segments:
        return "", []
    align_tokens_to_segments(text, segments, tokenizer)
    return text, segments


def _get_token_reward_output(text: str, llm_engine: LLM):
    vllm_outputs = llm_engine.encode([text])
    return vllm_outputs[0]


def _collect_assistant_scores(token_scores: torch.Tensor, segments: List[FormattedSegment]):
    token_values = []
    message_means = []
    seq_len = token_scores.shape[0]
    for segment in segments:
        if segment.role != 'assistant' or not segment.token_indices:
            continue
        valid_indices = [idx for idx in segment.token_indices if idx < seq_len]
        if not valid_indices:
            continue
        idx_tensor = torch.tensor(valid_indices, device=token_scores.device)
        seg_scores = token_scores.index_select(0, idx_tensor)
        if seg_scores.numel() == 0:
            continue
        token_values.append(seg_scores)
        message_means.append(seg_scores.mean())

    if token_values:
        concatenated = torch.cat(token_values)
    else:
        concatenated = torch.tensor([], device=token_scores.device)
    if message_means:
        message_tensor = torch.stack(message_means)
    else:
        message_tensor = torch.tensor([], device=token_scores.device)
    return concatenated, message_tensor


def score_trajectory_original(messages: List[Dict[str, Any]], llm_engine: LLM,
                              tokenizer: AutoTokenizer, **kwargs) -> float:
    text, segments = _prepare_text_and_segments(messages, tokenizer)
    if not text:
        return -2.0
    try:
        request_output = _get_token_reward_output(text, llm_engine)
        token_scores = request_output.outputs.data
        assistant_tokens, _ = _collect_assistant_scores(token_scores, segments)
        if assistant_tokens.numel() == 0:
            return token_scores[-1].item()
        return assistant_tokens[-1].item()
    except Exception as e:
        print(f"Error (Original Scheme - VLLM): {e}")
        return -2.0


def score_trajectory_scheme1(messages: List[Dict[str, Any]], llm_engine: LLM,
                             tokenizer: AutoTokenizer, k: int = 100, **kwargs) -> float:
    text, segments = _prepare_text_and_segments(messages, tokenizer)
    if not text:
        return -2.0
    try:
        request_output = _get_token_reward_output(text, llm_engine)
        token_scores = request_output.outputs.data
        assistant_tokens, _ = _collect_assistant_scores(token_scores, segments)
        if assistant_tokens.numel() == 0:
            tail_scores = token_scores[-k:]
        else:
            tail_scores = assistant_tokens[-k:]
        if tail_scores.numel() == 0:
            return -2.0
        return tail_scores.mean().item()
    except Exception as e:
        print(f"Error (Scheme 1 - VLLM): {e}")
        return -2.0


def score_trajectory_scheme2(messages: List[Dict[str, Any]], llm_engine: LLM,
                             tokenizer: AutoTokenizer, gamma: float = 0.99, **kwargs) -> float:
    text, segments = _prepare_text_and_segments(messages, tokenizer)
    if not text:
        return -2.0
    try:
        request_output = _get_token_reward_output(text, llm_engine)
        token_scores = request_output.outputs.data
        _, assistant_message_scores = _collect_assistant_scores(token_scores, segments)
        if assistant_message_scores.numel() == 0:
            return score_trajectory_original(messages, llm_engine, tokenizer, **kwargs)
        weights = torch.pow(gamma,
                            torch.arange(assistant_message_scores.numel() - 1, -1, -1,
                                         device=assistant_message_scores.device,
                                         dtype=torch.float32))
        weighted_sum = (assistant_message_scores * weights).sum()
        return (weighted_sum / weights.sum()).item()
    except Exception as e:
        print(f"Error (Scheme 2 - VLLM): {e}")
        return -2.0


def score_trajectory_scheme3(messages: List[Dict[str, Any]], llm_engine: LLM,
                             tokenizer: AutoTokenizer, **kwargs) -> float:
    text, segments = _prepare_text_and_segments(messages, tokenizer)
    if not text:
        return -2.0
    try:
        request_output = _get_token_reward_output(text, llm_engine)
        token_scores = request_output.outputs.data
        assistant_tokens, _ = _collect_assistant_scores(token_scores, segments)
        target_scores = assistant_tokens if assistant_tokens.numel() else token_scores
        weights = torch.nn.functional.softmax(target_scores, dim=0)
        return (target_scores * weights).sum().item()
    except Exception as e:
        print(f"Error (Scheme 3 - VLLM): {e}")
        return -2.0


# --- 3. 批量处理、打分和选择最优（加入断点续写逻辑 + 手动截断） ---

def batch_score_and_select_best(llm_engine, tokenizer, input_file, output_file, scoring_function):
    if not os.path.exists(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在。")
        return

    print(f"\n开始处理文件 '{input_file}' -> '{output_file}'...")

    # 统计用变量（会同时包含“历史已经处理过的实例”和本次新处理的实例）
    total_instances_processed = 0
    resolved_instances_count = 0

    # ---------- 1. 断点续写：读取已有输出，恢复 processed_ids 和统计 ----------
    processed_ids = set()
    out_mode = 'w'  # 默认从头写；如果检测到已有文件，则改成 'a'

    if os.path.exists(output_file):
        print(f"检测到已有输出文件 '{output_file}'，尝试从中恢复进度...")
        with open(output_file, 'r', encoding='utf-8') as f_out_old:
            for line in f_out_old:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    # 如果有损坏行，跳过，不影响整体
                    continue

                instance_id = obj.get("instance_id")
                if instance_id is None:
                    continue
                processed_ids.add(instance_id)

                # 恢复历史统计信息
                total_instances_processed += 1
                if obj.get("best_run_resolved"):
                    resolved_instances_count += 1

        out_mode = 'a'
        print(f"已从历史结果中恢复 {len(processed_ids)} 个已处理实例。")
        print(f"历史统计：已处理实例数 = {total_instances_processed}，"
              f"其中最优轨迹已解决的实例数 = {resolved_instances_count}")
    else:
        print("未检测到已有输出文件，将从头开始处理。")

    # ---------- 2. 预计算输入总行数（仅用于 tqdm 进度条） ----------
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
    except Exception as e:
        print(f"无法读取行数: {e}")
        total_lines = 0

    # ---------- 3. 正式遍历输入文件，跳过已处理的 instance_id ----------
    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, out_mode, encoding='utf-8') as f_out:

        line_idx = 0
        for line in tqdm(f_in, total=total_lines, desc="Scoring Instances with VLLM"):
            line_idx += 1
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告：第 {line_idx} 行不是合法 JSON，已跳过。内容: {line[:200]}")
                continue

            instance_id = data.get("instance_id")
            if not instance_id:
                continue

            # 断点续写关键逻辑：如果这个 instance_id 已经在输出文件中出现过，直接跳过
            if instance_id in processed_ids:
                continue

            run_data = {}
            for key, value in data.items():
                if key.startswith("run_"):
                    messages = value.get("funccalloff_messages")
                    is_resolved = value.get("resolved", False)

                    if not messages:
                        score = -10.0
                    else:
                        score = scoring_function(messages, llm_engine, tokenizer)

                    run_data[key] = {"score": score, "resolved": is_resolved}

            best_run_id = None
            best_score = -float('inf')
            best_run_is_resolved = None

            if run_data:
                # 选出得分最高的 run
                best_run_id, best_run_info = max(run_data.items(), key=lambda item: item[1]["score"])
                best_score = best_run_info["score"]
                best_run_is_resolved = best_run_info["resolved"]

                # 更新全局统计：这里包括本次新处理的实例（历史的已在前面恢复）
                total_instances_processed += 1
                if best_run_is_resolved:
                    resolved_instances_count += 1

            # 输出当前 instance 的所有 run 得分 + 最优信息
            output_line = {"instance_id": instance_id}
            for run_id, info in run_data.items():
                output_line[f"{run_id}_score"] = info["score"]
            output_line["best_run_id"] = best_run_id
            output_line["best_score"] = best_score
            output_line["best_run_resolved"] = best_run_is_resolved

            f_out.write(json.dumps(output_line, ensure_ascii=False) + '\n')

            # 为安全起见，偶尔 flush 一下，减少意外中断时的数据丢失
            if total_instances_processed % 100 == 0:
                f_out.flush()

    # ---------- 4. 打印最终统计（包含历史 + 本次） ----------
    print(f"\n处理完成！结果已保存到 '{output_file}'。")

    print("\n--- 最终统计（包含历史 + 本次） ---")
    if total_instances_processed > 0:
        resolved_rate = (resolved_instances_count / total_instances_processed) * 100
        print(f"总共处理的实例数: {total_instances_processed}")
        print(f"被评为最优且已解决的实例数: {resolved_instances_count}")
        print(f"基于 Critic 模型选择的最优轨迹解决率: {resolved_rate:.2f}%")
    else:
        print("没有处理任何有效的实例，无法计算解决率。")


# --- 主执行入口 ---

if __name__ == "__main__":
    # 想跑哪些方案就写在这里，支持重复运行 + 断点续写
    schemes_to_run = [0, 1, 2, 3]
    schemes = {
        0: ("Original", score_trajectory_original),
        1: ("Scheme 1: Tail Average", score_trajectory_scheme1),
        2: ("Scheme 2: Discounted Step-Ends", score_trajectory_scheme2),
        3: ("Scheme 3: Attention-Weighted Average", score_trajectory_scheme3),
    }

    for scheme_id in schemes_to_run:
        if scheme_id not in schemes:
            print(f"警告: 无效的方案ID '{scheme_id}'，已跳过。")
            continue
        scheme_name, score_func = schemes[scheme_id]
        print(f"\n\n{'=' * 30}")
        print(f"   开始执行方案 {scheme_id}: {scheme_name} (VLLM)")
        print(f"{'=' * 30}\n")
        output_file = OUTPUT_FILE_TEMPLATE.format(scheme=scheme_id)
        batch_score_and_select_best(llm_engine, tokenizer, INPUT_FILE, output_file, score_func)
        gc.collect()

    print("\n\n所有指定的打分方案均已执行完毕！")
