#!/bin/bash

################## jierun data
part="p1_749"
# part="p2_791"
# part="p3_690"
# part="p4_747"
# part="p5_674"
# part="p6_691"
resolved_file="/scratch/ywxzml3j/user30/code/OpenHands/evaluation/evaluation_outputs/outputs/autocode07__swerebench_${part}-train/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct_maxiter_100_N_v0.53.0-aci0.3.2-no-hint-intViewRange-run_1/Qwen3-Coder-480B-A35B-Instruct_maxiter_100_N_v0.53.0-aci0.3.2-no-hint-intViewRange-run_1.swerebench_${part}.json"
trajectory_path="/scratch/ywxzml3j/user30/code/OpenHands/evaluation/evaluation_outputs/outputs/autocode07__swerebench_${part}-train/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct_maxiter_100_N_v0.53.0-aci0.3.2-no-hint-intViewRange-run_1/llm_completions"
save_file_path="/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/rebench_${part}_<resolved_ids>_<effective_num>_incontext<add_in_context_learning_example>_funccalloff.json"


################## chaofan data
# resolved_file="/scratch/ywxzml3j/user23/to_jierun/logs/run_evaluation/generated-bug-inst-2508__Qwen3-Coder-480B-A35B-Instruct_maxiter_100_N_v0.53.0-no-hint-run_2/report.json"
# trajectory_path="/scratch/ywxzml3j/user30/code/OpenHands/evaluation/evaluation_outputs/outputs/autocode07__generated-bug-inst-2508-train_v0_invalid_view_range/CodeActAgent/Qwen3-Coder-480B-A35B-Instruct_maxiter_100_N_v0.53.0-no-hint-run_1/llm_completions"
# save_file_path="/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/autocode07__generated-bug-inst-2508_<resolved_ids>_<effective_num>_incontext<add_in_context_learning_example>_funccalloff.json"

# resolved_file="/home/ywxzml3j/ywxzml3juser23/OpenHands/evaluation/evaluation_outputs/outputs/autocode07__0911_swerebench_p1_749_inst_with_difficulties-train/CodeActAgent/Qwen3-Coder-30B-A3B-Instruct_maxiter_100_N_v0.53.0-no-hint-run_1/report.json"
# trajectory_path="/home/ywxzml3j/ywxzml3juser23/OpenHands/evaluation/evaluation_outputs/outputs/autocode07__0911_swerebench_p1_749_inst_with_difficulties-train/CodeActAgent/Qwen3-Coder-30B-A3B-Instruct_maxiter_100_N_v0.53.0-no-hint-run_1/llm_completions"
# save_file_path="/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/autocode07__0911_swerebench_p1_749_inst_with_difficulties_<resolved_ids>_<effective_num>_incontext<add_in_context_learning_example>_funccalloff.json"

# resolved_file="/home/ywxzml3j/ywxzml3juser23/OpenHands/evaluation/evaluation_outputs/outputs/autocode07__SWE-smith-p12345-train/CodeActAgent/Qwen3-Coder-30B-A3B-Instruct_maxiter_100_N_v0.53.0-no-hint-run_1/report.json"
# trajectory_path="/home/ywxzml3j/ywxzml3juser23/OpenHands/evaluation/evaluation_outputs/outputs/autocode07__SWE-smith-p12345-train/CodeActAgent/Qwen3-Coder-30B-A3B-Instruct_maxiter_100_N_v0.53.0-no-hint-run_1/llm_completions"
# save_file_path="/home/ywxzml3j/ywxzml3juser57/LLaMA-Factory/data/autocode07__SWE-smith-p12345_<resolved_ids>_<effective_num>_incontext<add_in_context_learning_example>_funccalloff.json"


python convert_openhands_to_funccalloff.py \
    --resolved_file $resolved_file \
    --trajectory_path $trajectory_path \
    --save_file_path $save_file_path