import os
import json
import torch
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
import gc

# --- 配置 ---
MODEL_PATH = "/scratch/ywxzml3j/user72/models/openhands-critic-32b-exp-20250417"
INPUT_FILE = "tts_funccalloff_all.json"
OUTPUT_FILE_TEMPLATE = "scored_results_scheme_hf_{scheme}.jsonl"
MAX_LENGTH = 30720

# --- 1. 加载模型和 Tokenizer (只需执行一次) ---

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ===== 关键优化: 设置从末尾向前截断 =====
tokenizer.truncation_side = 'left'
print(f"Tokenizer configured with truncation_side='{tokenizer.truncation_side}'")


print(f"Loading model '{MODEL_PATH}' with device_map='auto'...")
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True
)
model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
print("Model loaded successfully.")


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


def map_segments_to_tokens(segments: List[FormattedSegment], offsets: List[List[int]]) -> None:
    if not segments:
        return
    seg_idx = 0
    for token_idx, (start, end) in enumerate(offsets):
        if seg_idx >= len(segments):
            break
        if start == end:
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


def _prepare_text_and_segments(messages: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> tuple[str, List[FormattedSegment]]:
    text, segments = build_segments_from_messages(messages, tokenizer, MAX_LENGTH)
    if not text or not segments:
        return "", []
    return text, segments


def _tokenize_with_offsets(text: str, tokenizer: AutoTokenizer):
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
    )
    offsets = encoding.pop("offset_mapping").squeeze(0).tolist()
    return encoding, offsets


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


def score_trajectory_original(messages: List[Dict[str, Any]], model, tokenizer, **kwargs) -> float:
    text, segments = _prepare_text_and_segments(messages, tokenizer)
    if not text:
        return -2.0
    try:
        encoding, offsets = _tokenize_with_offsets(text, tokenizer)
        map_segments_to_tokens(segments, offsets)
        inputs = {k: v.to(model.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze(0).squeeze(-1)
        attention_mask = inputs['attention_mask'].squeeze(0)
        valid_len = int(attention_mask.sum().item())
        token_scores = logits[:valid_len]
        assistant_tokens, _ = _collect_assistant_scores(token_scores, segments)
        if assistant_tokens.numel() == 0:
            final_score = token_scores[-1].item()
        else:
            final_score = assistant_tokens[-1].item()
        del inputs, outputs, logits, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return final_score
    except Exception as e:
        print(f"Error (Original Scheme): {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return -2.0

def score_trajectory_scheme1(messages: List[Dict[str, Any]], model, tokenizer,
                             k: int = 100, **kwargs) -> float:
    text, segments = _prepare_text_and_segments(messages, tokenizer)
    if not text:
        return -2.0
    try:
        encoding, offsets = _tokenize_with_offsets(text, tokenizer)
        map_segments_to_tokens(segments, offsets)
        inputs = {k: v.to(model.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze(0).squeeze(-1)
        attention_mask = inputs['attention_mask'].squeeze(0)
        valid_len = int(attention_mask.sum().item())
        token_scores = logits[:valid_len]
        assistant_tokens, _ = _collect_assistant_scores(token_scores, segments)
        if assistant_tokens.numel() == 0:
            tail_scores = token_scores[-k:]
        else:
            tail_scores = assistant_tokens[-k:]
        if tail_scores.numel() == 0:
            final_score = -2.0
        else:
            final_score = tail_scores.mean().item()
        del inputs, outputs, logits, attention_mask, tail_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return final_score
    except Exception as e:
        print(f"Error (Scheme 1): {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return -2.0

def score_trajectory_scheme2(messages: List[Dict[str, Any]], model, tokenizer,
                             gamma: float = 0.99, **kwargs) -> float:
    text, segments = _prepare_text_and_segments(messages, tokenizer)
    if not text:
        return -2.0
    try:
        encoding, offsets = _tokenize_with_offsets(text, tokenizer)
        map_segments_to_tokens(segments, offsets)
        inputs = {k: v.to(model.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze(0).squeeze(-1)
        attention_mask = inputs['attention_mask'].squeeze(0)
        valid_len = int(attention_mask.sum().item())
        token_scores = logits[:valid_len]
        _, assistant_message_scores = _collect_assistant_scores(token_scores, segments)
        if assistant_message_scores.numel() == 0:
            final_score = score_trajectory_original(messages, model, tokenizer, **kwargs)
        else:
            discounts = torch.pow(
                gamma,
                torch.arange(assistant_message_scores.numel() - 1, -1, -1,
                             device=assistant_message_scores.device,
                             dtype=torch.float32),
            )
            weighted_sum = (assistant_message_scores * discounts).sum()
            final_score = (weighted_sum / discounts.sum()).item()
        del inputs, outputs, logits, attention_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return final_score
    except Exception as e:
        print(f"Error (Scheme 2): {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return -2.0

def score_trajectory_scheme3(messages: List[Dict[str, Any]], model, tokenizer, **kwargs) -> float:
    text, segments = _prepare_text_and_segments(messages, tokenizer)
    if not text:
        return -2.0
    try:
        encoding, offsets = _tokenize_with_offsets(text, tokenizer)
        map_segments_to_tokens(segments, offsets)
        inputs = {k: v.to(model.device) for k, v in encoding.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits.squeeze(0).squeeze(-1)
        attention_mask = inputs['attention_mask'].squeeze(0)
        valid_len = int(attention_mask.sum().item())
        token_scores = logits[:valid_len]
        assistant_tokens, _ = _collect_assistant_scores(token_scores, segments)
        target_scores = assistant_tokens if assistant_tokens.numel() else token_scores
        weights = torch.nn.functional.softmax(target_scores, dim=0)
        final_score = (target_scores * weights).sum().item()
        del inputs, outputs, logits, attention_mask, target_scores, weights
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return final_score
    except Exception as e:
        print(f"Error (Scheme 3): {e}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return -2.0


# --- 3. 批量处理、打分和选择最优 (代码无变化) ---

def batch_score_and_select_best(model, tokenizer, input_file, output_file, scoring_function):
    """
    读取聚合后的jsonl文件，使用指定的scoring_function为每个轨迹打分，
    选出最优的run，并计算最终的解决率。
    """
    if not os.path.exists(input_file):
        print(f"错误：输入文件 '{input_file}' 不存在。")
        return

    print(f"开始处理文件 '{input_file}' -> '{output_file}'...")
    
    total_instances_processed = 0
    resolved_instances_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for line in f)
    except Exception as e:
        print(f"无法读取行数: {e}")
        total_lines = 0

    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total_lines, desc="Scoring Instances"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"警告：跳过无效的JSON行: {line.strip()}")
                continue
            
            instance_id = data.get("instance_id")
            if not instance_id:
                continue

            run_data = {}
            for key, value in data.items():
                if key.startswith("run_"):
                    messages = value.get("funccalloff_messages")
                    is_resolved = value.get("resolved", False)

                    if not messages:
                        score = -10.0
                    else:
                        score = scoring_function(messages, model, tokenizer)

                    run_data[key] = {"score": score, "resolved": is_resolved}
            
            best_run_id = None
            best_score = -float('inf')
            best_run_is_resolved = None
            
            if run_data:
                best_run_id, best_run_info = max(run_data.items(), key=lambda item: item[1]["score"])
                best_score = best_run_info["score"]
                best_run_is_resolved = best_run_info["resolved"]

                total_instances_processed += 1
                if best_run_is_resolved:
                    resolved_instances_count += 1
            
            output_line = {"instance_id": instance_id}
            for run_id, info in run_data.items():
                output_line[f"{run_id}_score"] = info["score"]
            output_line["best_run_id"] = best_run_id
            output_line["best_score"] = best_score
            output_line["best_run_resolved"] = best_run_is_resolved
            
            f_out.write(json.dumps(output_line, ensure_ascii=False) + '\n')

    print(f"\n处理完成！结果已保存到 '{output_file}'。")

    print("\n--- 最终统计 ---")
    if total_instances_processed > 0:
        resolved_rate = (resolved_instances_count / total_instances_processed) * 100
        print(f"总共处理的实例数: {total_instances_processed}")
        print(f"被评为最优且已解决的实例数: {resolved_instances_count}")
        print(f"基于 Critic 模型选择的最优轨迹解决率: {resolved_rate:.2f}%")
    else:
        print("没有处理任何有效的实例，无法计算解决率。")


# --- 主执行入口 (已修改为循环执行) ---
if __name__ == "__main__":
    # =========================================================
    # == 在这里定义所有需要执行的打分方案列表 ==
    schemes_to_run = [2, 0, 1, 3]
    # =========================================================

    # 打分方案的定义字典保持不变
    schemes = {
        0: ("Original", score_trajectory_original),
        1: ("Scheme 1: Tail Average", score_trajectory_scheme1),
        2: ("Scheme 2: Discounted Step-Ends", score_trajectory_scheme2),
        3: ("Scheme 3: Attention-Weighted Average", score_trajectory_scheme3),
    }

    # 循环执行指定的每个方案
    for scheme_id in schemes_to_run:
        if scheme_id not in schemes:
            print(f"警告: 无效的方案ID '{scheme_id}'，已跳过。请从 {list(schemes.keys())} 中选择。")
            continue

        scheme_name, score_func = schemes[scheme_id]
        
        # 打印清晰的分割线和当前执行的方案信息
        print(f"\n\n{'='*30}")
        print(f"   开始执行方案 {scheme_id}: {scheme_name}")
        print(f"{'='*30}\n")
        
        # 根据方案ID动态生成输出文件名
        output_file = OUTPUT_FILE_TEMPLATE.format(scheme=scheme_id)
        
        # 调用核心处理函数，传入当前方案对应的打分函数
        batch_score_and_select_best(model, tokenizer, INPUT_FILE, output_file, score_func)

        # 在每个方案结束后进行一次垃圾回收，保持环境清洁
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n\n所有指定的打分方案均已执行完毕！")
