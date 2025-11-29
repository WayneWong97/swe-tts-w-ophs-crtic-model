import os
import json
import glob
import re
from collections import defaultdict

# --- 配置 ---
# 输入文件所在的目录
INPUT_DIR = "/scratch/ywxzml3j/user72/tts/qwen3-8b-welltrained_logs_enriched"  # 假设文件在当前目录下
# 输出文件的路径
OUTPUT_FILE = "qwen3-8b-welltrained_consolidated_data_8.jsonl"
# 输入文件的匹配模式
FILE_PATTERN = "output_run_*.jsonl"

def calculate_and_print_pass_at_k(aggregated_data: dict, num_runs: int):
    """
    根据聚合数据计算并打印 pass@k 指标。

    pass@k 的定义是：对于任何一个问题（instance_id），
    在前 k 次尝试（run_1, ..., run_k）中，只要有至少一次成功（resolved=True），
    就认为该问题在 k 次尝试内“通过”了。
    最终的 pass@k 概率 = (在 k 次尝试内通过的问题总数) / (问题总数)。

    Args:
        aggregated_data (dict): 按 instance_id 聚合的数据。
        num_runs (int): 处理的总运行次数（文件数）。
    """
    if not aggregated_data or num_runs == 0:
        print("\n没有足够的数据来计算 pass@k。")
        return

    total_instances = len(aggregated_data)
    # pass_counts[i] 将存储 pass@(i+1) 的成功实例数
    pass_counts = [0] * num_runs

    # 遍历每一个问题实例
    for instance_id, runs_data in aggregated_data.items():
        # has_passed_so_far 标记从 run_1 到当前 run_k 是否有成功的
        has_passed_so_far = False
        # 依次检查 pass@1, pass@2, ..., pass@k
        for k in range(1, num_runs + 1):
            run_key = f"run_{k}"
            
            # 如果之前的 run 尚未成功，则检查当前 run
            if not has_passed_so_far:
                # 使用 .get() 链来安全地访问，避免 KeyError，并确保值为 True
                if runs_data.get(run_key, {}).get("resolved") is True:
                    has_passed_so_far = True
            
            # 如果在 k 次或更少的尝试中已经成功，则 pass@k 的计数器加一
            if has_passed_so_far:
                pass_counts[k-1] += 1

    print("\n--- Pass@k 统计 ---")
    print(f"基于 {total_instances} 个独立问题实例和 {num_runs} 次运行计算：")
    for k in range(1, num_runs + 1):
        count = pass_counts[k-1]
        rate = (count / total_instances) * 100 if total_instances > 0 else 0
        print(f"pass@{k}: {count}/{total_instances} = {rate:.2f}%")

def aggregate_run_outputs(input_dir: str, file_pattern: str, output_file: str):
    """
    聚合多个运行的输出文件到一个统一的jsonl文件中。
    新增功能：提取每个run的 'resolved' 状态，并计算最终的 pass@k 指标。

    Args:
        input_dir (str): 存放 output_run<k>.jsonl 文件的目录。
        file_pattern (str): 用于匹配输入文件的 glob 模式。
        output_file (str): 最终输出的 jsonl 文件名。
    """
    aggregated_data = defaultdict(dict)
    
    search_path = os.path.join(input_dir, file_pattern)
    file_paths = glob.glob(search_path)
    
    if not file_paths:
        print(f"警告：在目录 '{input_dir}' 中没有找到任何匹配 '{file_pattern}' 的文件。")
        return

    # 确定总的运行次数 k
    num_runs_found = len(file_paths)
    print(f"找到 {num_runs_found} 个文件进行处理...")

    for file_path in sorted(file_paths):
        match = re.search(r'run_(\d+)\.jsonl$', os.path.basename(file_path))
        if not match:
            print(f"跳过文件：{file_path} (无法提取 run number)")
            continue
        
        run_k = match.group(1)
        run_key = f"run_{run_k}"
        print(f"正在处理文件: {file_path} (作为 {run_key})...")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        
                        instance_id = data.get("instance_id")
                        if not instance_id:
                            print(f"  - 警告: 在文件 {file_path} 的某一行中缺少 'instance_id'，已跳过。")
                            continue

                        git_patch = data.get("test_result", {}).get("git_patch")
                        history = data.get("history")
                        
                        # 使用 .get() 链式调用来安全地访问嵌套字典
                        report = data.get("report", {})
                        resolved_status = data.get("resolved")
                        
                        trajectory_data = {
                            "patch": git_patch,
                            "trajectory": history,
                            "resolved": resolved_status
                        }
                        
                        aggregated_data[instance_id][run_key] = trajectory_data
                        
                    except json.JSONDecodeError:
                        print(f"  - 警告: 在文件 {file_path} 中遇到无效的 JSON 行，已跳过。")
        except IOError as e:
            print(f"错误：无法读取文件 {file_path}: {e}")

    # --- 写入到输出文件 ---
    if not aggregated_data:
        print("\n没有聚合到任何数据，处理结束。")
        return

    print(f"\n聚合完成。正在将 {len(aggregated_data)} 条记录写入到 {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for instance_id, runs_data in aggregated_data.items():
            output_line = {
                "instance_id": instance_id,
                **runs_data
            }
            f_out.write(json.dumps(output_line, ensure_ascii=False) + '\n')
    
    # === 新增功能：计算并打印 pass@k ===
    calculate_and_print_pass_at_k(aggregated_data, num_runs_found)
            
    print("\n处理完成！")


if __name__ == "__main__":
    aggregate_run_outputs(INPUT_DIR, FILE_PATTERN, OUTPUT_FILE)
