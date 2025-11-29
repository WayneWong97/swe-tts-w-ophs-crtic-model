import json
import os
import copy
import argparse
from fn_call_converter import convert_from_multiple_tool_calls_to_single_tool_call_messages, convert_fncall_messages_to_non_fncall_messages, FunctionCallConversionError


def get_json_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as fcc_file:
        json_list = json.load(fcc_file)
    return json_list


def find_latest_json_filename(folder_path):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"文件夹路径不存在: {folder_path}")
    
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"路径不是文件夹: {folder_path}")
    
    files = os.listdir(folder_path)
    
    # sort the json files
    files.sort()
    # last file as the key file
    if files:
        key_json_file = files[-1]
    else:
        key_json_file = None
    return key_json_file


def process_response_of_json_data(json_data):
    response = json_data['response']['choices'][0]['message']

    if response['content']:
        content = [{"type": "text", "text": response['content']}]
    else:
        content = []
    reformat_response = {
        'content': content,
        "role": response['role'],
        "tool_calls": response['tool_calls']
    }

    new_json_data = {
        "messages": json_data['messages'] + [reformat_response],
        "tools": json_data['kwargs']['tools']
    }
    return new_json_data


def convert_messages(messages: list[dict], tools: list[dict], add_in_context_learning_example: bool) -> list[dict]:
    messages = convert_from_multiple_tool_calls_to_single_tool_call_messages(messages, ignore_final_tool_result=True)
    message_copy = copy.deepcopy(messages)
    for message in message_copy:
        if message['content'] is None:
            message['content'] = ''
    try:
        new_messages = convert_fncall_messages_to_non_fncall_messages(
            message_copy, tools, add_in_context_learning_example=add_in_context_learning_example
        )
        return {'messages': [{'role': m['role'], 'content': m['content'][0]['text']} for m in new_messages]}
    
    except FunctionCallConversionError:
        print(f'Failed to convert messages: {messages}\nTools: {tools}')
        return None


def rv_task_management_task_tracker(json_data):
    task_management_string = "\n\n<TASK_MANAGEMENT>\n* You have access to the `task_tracker` tool to help you organize and monitor development work. Use this tool REGULARLY to maintain task visibility and provide users with clear progress updates. This tool is ESSENTIAL for systematic planning and decomposing complex development work into manageable components. Failing to use this tool for planning may result in overlooked requirements - which is unacceptable.\n* It is crucial that you update task status to \"done\" immediately upon completion of each work item. Do not accumulate multiple finished tasks before updating their status.\n* For complex, multi-phase development work, use `task_tracker` to establish a comprehensive plan with well-defined steps:\n  1. Begin by decomposing the overall objective into primary phases using `task_tracker`\n  2. Include detailed work items as necessary to break complex activities into actionable units\n  3. Update tasks to \"in_progress\" status when commencing work on them\n  4. Update tasks to \"done\" status immediately after completing each item\n  5. For each primary phase, incorporate additional work items as you identify new requirements\n  6. If you determine the plan requires substantial modifications, suggest revisions and obtain user confirmation before proceeding\n* Example workflow for debugging and resolution:\n  ```\n  User: \"Execute the test suite and resolve any validation failures\"\n  Assistant: I'm going to use the task_tracker tool to organize the following work items:\n  - Execute the test suite\n  - Resolve any validation failures\n  I'm now going to run the test suite using the terminal.\n  [After running tests and discovering 8 validation failures]\n  I found 8 validation failures that need attention. I'm going to use the task_tracker tool to add 8 specific items to the task list.\n  [Updating first task to in_progress]\n  Let me begin addressing the first validation issue...\n  [After resolving first failure]\n  The first validation issue has been resolved, let me mark that task as done and proceed to the second item...\n  ```\n* Example workflow for component development:\n  ```\n  User: \"Build a dashboard component that displays analytics data with interactive charts and filtering options\"\n  Assistant: I'll help you create an analytics dashboard with interactive charts and filtering. Let me first use the task_tracker tool to organize this development work.\n  Adding the following tasks to the tracker:\n  1. Analyze existing analytics data structure and requirements\n  2. Design dashboard layout and component architecture\n  3. Implement data visualization charts with interactivity\n  4. Create filtering and search functionality\n  5. Integrate components and perform testing\n  Let me start by examining the current analytics data structure to understand what we're working with...\n  [Assistant proceeds with implementation step by step, updating tasks to in_progress and done as work progresses]\n  ```\n</TASK_MANAGEMENT>"
    # Check if the first message is from the system and contains the task management instructions
    
    if json_data['messages'][0]['role'] == 'system':
        sysprompt = json_data['messages'][0]['content'][0]["text"] 
        # replace the task management part
        json_data['messages'][0]['content'][0]["text"] = sysprompt.replace(task_management_string, "")
    
    # raise if <TASK_MANAGEMENT> still in the message
    if "<TASK_MANAGEMENT>" in json_data['messages'][0]['content'][0]["text"]:
        raise ValueError("TASK_MANAGEMENT string still in the system prompt")
    
    # rv the task tracker tool calls
    tools = json_data["kwargs"]["tools"]
    json_data["kwargs"]["tools"] = [tool for tool in tools if tool["function"]["name"] != "task_tracker"]
    
    return json_data


def check_if_task_tracker_used(json_data) -> bool:
    messages = json_data['messages']
    for message in messages:
        if message['role'] == 'assistant' and message.get('tool_calls'):
            for tool_call in message['tool_calls']:
                if tool_call["function"]['name'] == 'task_tracker':
                    return True
    return False


def check_if_duplicated_role_message(json_data):
    for i in range(1, len(json_data['messages'])):
        if json_data['messages'][i]['role'] == json_data['messages'][i-1]['role']:
            return True
    return False



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolved_file', default=None, type=str)
    parser.add_argument('--trajectory_path', default=None, type=str)
    parser.add_argument('--save_file_path', default=None, type=str)
    parser.add_argument('--add_in_context_learning_example', action='store_true')
    args = parser.parse_args()


    if "user30" in args.resolved_file:
        resolved_ids = get_json_list(args.resolved_file)['resolved_ids']
    else:
        resolved_ids = get_json_list(args.resolved_file)['ids_resolved']

    all_json_data = []
    for resolved_id in resolved_ids:
        folder_path = os.path.join(args.trajectory_path, resolved_id)
        if find_latest_json_filename(folder_path):
            json_data = get_json_list(os.path.join(folder_path, find_latest_json_filename(folder_path)))
        else:
            print(f'Skipping {resolved_id} because no json file')
            continue

        new_json_data_0 = rv_task_management_task_tracker(json_data)
        
        if  check_if_task_tracker_used(new_json_data_0):
            print(f'Skipping {resolved_id} because task_tracker used')
            continue
        
        new_json_data_1 = process_response_of_json_data(new_json_data_0)
        new_json_data_2 = convert_messages(new_json_data_1['messages'], new_json_data_1['tools'], args.add_in_context_learning_example)

        if check_if_duplicated_role_message(new_json_data_2):
            print(f'Skipping {resolved_id} because duplicated role message')
            continue

        all_json_data.append(new_json_data_2)

    # replace <resolved_ids> <effective_num> <add_in_context_learning_example> in save_file_path
    save_file_path = args.save_file_path.replace("<resolved_ids>", str(len(resolved_ids)))
    save_file_path = save_file_path.replace("<effective_num>", str(len(all_json_data)))
    save_file_path = save_file_path.replace("<add_in_context_learning_example>", str(args.add_in_context_learning_example))

    with open(save_file_path, "w", encoding="utf-8") as f:
        json.dump(all_json_data, f, ensure_ascii=False, indent=4)
    print(f"Save {len(all_json_data)} data to {save_file_path}")