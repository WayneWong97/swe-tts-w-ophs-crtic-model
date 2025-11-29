import os
import json

# --- 配置 ---

# 1. 存放 report_run_k.json 文件的目录
REPORTS_DIR = "/scratch/ywxzml3j/user72/tts/qwen3-8b-welltrained_reports/"

# 2. 存放原始 output_run_k.jsonl 文件的目录
#    (假设它们位于 /scratch/ywxzml3j/user72/tts/logs/)
ORIGINAL_OUTPUTS_DIR = "/scratch/ywxzml3j/user72/tts/qwen3-8b-welltrained_logs/"

# 3. 存放处理后、添加了 'resolved' 字段的新 output 文件的目录
#    (脚本会自动创建这个目录)
ENRICHED_OUTPUTS_DIR = "/scratch/ywxzml3j/user72/tts/qwen3-8b-welltrained_logs_enriched/"

# 4. 要处理的运行总数
NUM_RUNS = 8


def enrich_output_files(reports_dir, original_outputs_dir, enriched_outputs_dir, num_runs):
    """
    使用 report 文件中的 'resolved_ids' 信息，
    为 output.jsonl 文件中的每一行添加 'resolved' 字段。
    """
    
    # 确保输出目录存在
    print(f"确保输出目录存在: {enriched_outputs_dir}")
    os.makedirs(enriched_outputs_dir, exist_ok=True)
    
    print("\n开始处理文件...")
    
    # 循环处理每一对 run 文件 (run_1, run_2, ...)
    for k in range(1, num_runs + 1):
        
        report_filename = f"report_run_{k}.json"
        output_filename = f"output_run_{k}.jsonl"
        
        report_filepath = os.path.join(reports_dir, report_filename)
        original_output_filepath = os.path.join(original_outputs_dir, output_filename)
        enriched_output_filepath = os.path.join(enriched_outputs_dir, output_filename)
        
        print(f"\n--- 正在处理 Run {k} ---")
        
        # --- 步骤 1: 加载 resolved_ids ---
        try:
            with open(report_filepath, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
            
            # 使用 .get() 安全地获取 'resolved_ids'，如果不存在则返回空列表
            resolved_ids_list = report_data.get("resolved_ids", [])
            # 将列表转换为 set，以实现 O(1) 的平均查找时间复杂度
            resolved_ids_set = set(resolved_ids_list)
            
            print(f"  - 从 {report_filename} 加载了 {len(resolved_ids_set)} 个已解决的 ID。")
            
        except FileNotFoundError:
            print(f"  - 错误: 报告文件未找到: {report_filepath}")
            print(f"  - 跳过 Run {k} 的处理。")
            continue
        except json.JSONDecodeError:
            print(f"  - 错误: 无法解析 JSON 文件: {report_filepath}")
            print(f"  - 跳过 Run {k} 的处理。")
            continue

        # --- 步骤 2: 处理 output.jsonl 文件并添加字段 ---
        try:
            lines_processed = 0
            # 同时打开输入和输出文件
            with open(original_output_filepath, 'r', encoding='utf-8') as f_in, \
                 open(enriched_output_filepath, 'w', encoding='utf-8') as f_out:
                
                for line in f_in:
                    try:
                        # 解析每一行 JSON
                        data = json.loads(line)
                        instance_id = data.get("instance_id")
                        
                        if instance_id:
                            # 核心逻辑：检查 instance_id 是否在 set 中
                            if instance_id in resolved_ids_set:
                                data['resolved'] = True
                            else:
                                data['resolved'] = False
                        else:
                            # 如果某行没有 instance_id，也给一个默认值
                            data['resolved'] = False

                        # 将更新后的数据写回新文件
                        f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                        lines_processed += 1
                        
                    except json.JSONDecodeError:
                        # 如果某一行不是有效的 JSON，直接原样写入并打印警告
                        print(f"    - 警告: 发现无效的 JSON 行，已原样复制。内容: {line.strip()}")
                        f_out.write(line)
            
            print(f"  - 成功处理 {lines_processed} 行，结果已保存到 {enriched_output_filepath}")

        except FileNotFoundError:
            print(f"  - 错误: 输出文件未找到: {original_output_filepath}")
            print(f"  - 无法完成 Run {k} 的处理。")
            continue


if __name__ == "__main__":
    enrich_output_files(REPORTS_DIR, ORIGINAL_OUTPUTS_DIR, ENRICHED_OUTPUTS_DIR, NUM_RUNS)
    print("\n\n所有处理已完成！")
