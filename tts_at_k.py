import json
import os
import re
from collections import defaultdict
import pandas as pd

# --- 配置 ---
# 自动查找所有打分策略的输出文件
OUTPUT_FILE_PATTERN = "scored_results_scheme_8_*.jsonl"
# 原始数据文件，用于获取每个run的真实“resolved”状态
CONSOLIDATED_FILE = "qwen3-8b-welltrained_consolidated_data_8.jsonl"
# 方案名称映射，用于美化输出
SCHEME_NAMES = {
    0: "Original",
    1: "Scheme 1: Tail Average",
    2: "Scheme 2: Discounted Step-Ends",
    3: "Scheme 3: Attention-Weighted Average",
}


def analyze_cumulative_selection(score_file_path: str, consolidated_data: dict):
    """
    根据用户的精确逻辑计算TTS@k。
    TTS@k: 考虑前k个run (run_1...run_k)作为一个选择池，
    选出其中分数最高的一个run，然后检查这个run是否成功。
    该指标衡量模型在信息逐渐增多时，做出正确选择的能力。
    """
    if not os.path.exists(score_file_path):
        print(f"错误: 文件 '{score_file_path}' 不存在。")
        return None, 0, 0

    # tts_at_k_successes[k] 存储了在使用前k个run作为选择池时，选出的最优run是成功的实例数量
    tts_at_k_successes = defaultdict(int)
    total_instances = 0
    max_k_for_scheme = 0

    print(f"\n--- 正在分析评分文件: {os.path.basename(score_file_path)} ---")

    with open(score_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                score_data = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            instance_id = score_data.get("instance_id")
            if not instance_id or instance_id not in consolidated_data:
                continue
            
            total_instances += 1
            original_instance_data = consolidated_data[instance_id]

            # 1. 为当前实例收集所有 run 的完整信息（序号, 分数, 解决状态）
            all_runs_info = []
            for key, value in original_instance_data.items():
                match = re.match(r'run_(\d+)', key)
                if match and isinstance(value, dict):
                    run_number = int(match.group(1))
                    score = score_data.get(f"{key}_score", -float('inf'))
                    is_resolved = value.get("resolved", False)
                    all_runs_info.append({
                        "run_number": run_number,
                        "score": score,
                        "resolved": is_resolved
                    })

            # 2. 必须严格按照 run 的原始序号排序，以构建逐步扩大的选择池
            all_runs_info.sort(key=lambda x: x["run_number"])
            
            num_runs = len(all_runs_info)
            if num_runs == 0:
                continue
            max_k_for_scheme = max(max_k_for_scheme, num_runs)

            # 3. 核心逻辑：迭代计算 TTS@1, TTS@2, ...
            for k in range(1, num_runs + 1):
                # a. 定义当前的选择池（从 run_1 到 run_k）
                current_pool = all_runs_info[:k]
                
                # b. 从当前池中，根据分数选出最优的一个 run
                best_choice_in_pool = max(current_pool, key=lambda x: x["score"])
                
                # c. 检查这个被选中的 run 是否解决了问题
                if best_choice_in_pool["resolved"]:
                    # d. 如果是，则这个实例在 TTS@k 上是成功的
                    tts_at_k_successes[k] += 1

    if total_instances == 0:
        return None, 0, 0

    # 4. 计算并返回最终的成功率
    tts_at_k_rates = {k: (count / total_instances) * 100 for k, count in tts_at_k_successes.items()}
    
    return tts_at_k_rates, total_instances, max_k_for_scheme


def load_consolidated_data(input_file: str):
    """将原始的 consolidated jsonl 文件加载到字典中以便快速查找真实状态。"""
    if not os.path.exists(input_file):
        print(f"致命错误: 无法找到计算所需的基础文件 '{input_file}'。")
        return None
        
    print(f"正在从 '{input_file}' 加载原始数据...")
    data_map = {}
    with open(input_file, 'r', encoding='utf-8') as f:
        for record in f:
            try:
                data = json.loads(record)
                instance_id = data.get("instance_id")
                if instance_id:
                    data_map[instance_id] = data
            except json.JSONDecodeError:
                continue
    print(f"加载了 {len(data_map)} 条实例数据。")
    return data_map

# --- 主执行入口 ---
if __name__ == "__main__":
    # 步骤1: 加载一次真实数据
    consolidated_data = load_consolidated_data(CONSOLIDATED_FILE)
    if not consolidated_data:
        exit(1)

    # 步骤2: 找到所有需要分析的打分文件
    import glob
    files_to_process = sorted(glob.glob(OUTPUT_FILE_PATTERN))
    if not files_to_process:
        print(f"错误: 在当前目录下没有找到匹配 '{OUTPUT_FILE_PATTERN}' 的文件。")
        exit()
        
    print(f"发现了 {len(files_to_process)} 个打分文件: {files_to_process}")

    # 步骤3: 循环处理每个文件，并存储结果
    all_results = {}
    max_k_overall = 0
    
    for file_path in files_to_process:
        match = re.search(r'scheme_8_(\d+)\.jsonl', file_path)
        if not match:
            continue
        
        scheme_id = int(match.group(1))
        scheme_name = SCHEME_NAMES.get(scheme_id, f"Scheme {scheme_id}")

        rates, total_instances, max_k = analyze_cumulative_selection(file_path, consolidated_data)
        
        if rates:
            all_results[scheme_name] = rates
            max_k_overall = max(max_k_overall, max_k)
            print(f"分析完成。共处理 {total_instances} 个实例。")
            for k in sorted(rates.keys()):
                print(f"  - TTS@{k}: {rates[k]:.2f}%")

    # 步骤4: 汇总所有策略的结果，以表格形式清晰展示
    if all_results and max_k_overall > 0:
        print("\n\n" + "="*70)
        print("  Critic Model Cumulative Selection Performance (TTS@k Success Rate %)")
        print("="*70)
        
        df_data = []
        schemes_sorted = sorted(all_results.keys())
        for scheme_name in schemes_sorted:
            row = {"Scheme": scheme_name}
            rates = all_results[scheme_name]
            for k in range(1, max_k_overall + 1):
                # 如果某个方案的最大k值不足，我们不能像之前一样填充
                # 因为TTS@k的结果是独立的，TTS@4的结果不等于TTS@5的结果
                rate = rates.get(k, 0.0) # 如果没有那个k值，说明没有那么多轮次
                row[f"TTS@{k}"] = f"{rate:.2f}"
            df_data.append(row)

        df = pd.DataFrame(df_data)
        df.set_index("Scheme", inplace=True)
        
        print(df.to_string())
        
        print("\n" + "="*70)
        print("分析完成！该表格显示了当选择池限定在前k轮时，模型选出的最优解的成功率。")
