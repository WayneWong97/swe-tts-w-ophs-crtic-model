import argparse
import copy
import json
from typing import Any, Dict, List, Optional

from convert_openhands_to_funccalloff import convert_messages


# Lightweight tool specs so convert_messages can reuse the same prompt logic.
DEFAULT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "description": "Run shell commands inside the repository workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command to run (can contain pipes and logical operators).",
                    },
                    "security_risk": {
                        "type": "string",
                        "description": "Qualitative risk assessment reported to the user.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_editor",
            "description": "Read, create, or edit files using a string-replace style interface.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute file path."},
                    "command": {
                        "type": "string",
                        "description": "Operation to run (read/create/str_replace).",
                    },
                    "start": {"type": "integer", "description": "Optional start line."},
                    "end": {"type": "integer", "description": "Optional end line."},
                    "file_text": {
                        "type": "string",
                        "description": "File contents for create commands.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Original text to replace when command=str_replace.",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text when command=str_replace.",
                    },
                },
                "required": ["path", "command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Log structured reasoning that is hidden from the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {"type": "string", "description": "Detailed reasoning."}
                },
                "required": ["thought"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Return the final answer to the user and end the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Final response summarizing the work.",
                    }
                },
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task_tracker",
            "description": "Plan, update, and review task tracker items during the session.",
            "parameters": {
                "type": "object",
                # No explicit properties so that any payload shape passes validation.
            },
        },
    },
]


def _text_content(text: str) -> List[Dict[str, str]]:
    return [{"type": "text", "text": text}]


def build_messages_from_trajectory(
    trajectory: List[Dict[str, Any]]
) -> Optional[List[Dict[str, Any]]]:
    messages: List[Dict[str, Any]] = []
    steps = sorted(trajectory, key=lambda item: item.get("id", 0))

    for step in steps:
        action = step.get("action")
        source = step.get("source")
        metadata = step.get("tool_call_metadata")

        if action == "system":
            messages.append({"role": "system", "content": _text_content(step.get("message", ""))})
            continue

        if source == "user" and action == "message":
            messages.append({"role": "user", "content": _text_content(step.get("message", ""))})
            continue

        if source == "user":
            # Skip recall/other helper actions initiated by the harness.
            continue

        if action == "message" and source == "agent":
            messages.append({"role": "assistant", "content": _text_content(step.get("message", ""))})
            continue

        if action and metadata:
            model_message = copy.deepcopy(metadata["model_response"]["choices"][0]["message"])
            # Some providers return None content; normalize to empty string for downstream logic.
            content = model_message.get("content")
            if content is None:
                model_message["content"] = _text_content("")
            elif isinstance(content, str):
                model_message["content"] = _text_content(content)
            messages.append(model_message)
            continue

        if metadata and step.get("cause") is not None:
            content = step.get("content") or step.get("message") or ""
            tool_message: Dict[str, Any] = {
                "role": "tool",
                "name": metadata.get("function_name", "tool"),
                "content": _text_content(content),
            }
            tool_call_id = metadata.get("tool_call_id")
            if tool_call_id:
                tool_message["tool_call_id"] = tool_call_id
            messages.append(tool_message)

    return messages if messages else None


def remove_consecutive_duplicate_roles(
    messages: List[Dict[str, Any]]
) -> tuple[List[Dict[str, Any]], int]:
    cleaned_messages: List[Dict[str, Any]] = []
    removed = 0
    last_role: Optional[str] = None
    for message in messages:
        role = message.get("role")
        if role == last_role:
            removed += 1
            continue
        cleaned_messages.append(message)
        last_role = role
    return cleaned_messages, removed


def convert_trajectory_to_funccall_messages(
    trajectory: List[Dict[str, Any]],
    add_in_context_learning_example: bool,
) -> tuple[Optional[List[Dict[str, Any]]], Optional[str], int]:
    # Guard: trajectory 可能为 None/非列表，统一兜底为空列表
    if not trajectory or not isinstance(trajectory, list):
        return None, "empty_trajectory", 0

    messages = build_messages_from_trajectory(trajectory)
    if not messages:
        return None, "empty_trajectory", 0

    converted = convert_messages(messages, DEFAULT_TOOLS, add_in_context_learning_example)
    if not converted:
        return None, "convert_messages_failed", 0

    cleaned_messages, removed_duplicates = remove_consecutive_duplicate_roles(
        converted["messages"]
    )
    return cleaned_messages, None, removed_duplicates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", required=True, type=str)
    parser.add_argument("--save_file_path", required=True, type=str)
    parser.add_argument("--add_in_context_learning_example", action="store_true")
    args = parser.parse_args()

    total_instances = 0
    total_runs = 0
    converted_runs = 0
    failure_reasons: Dict[str, int] = {}

    with open(args.jsonl_path, "r", encoding="utf-8") as infile, open(
        args.save_file_path, "w", encoding="utf-8"
    ) as outfile:
        for raw_line in infile:
            if not raw_line.strip():
                continue

            total_instances += 1
            entry = json.loads(raw_line)
            entry_copy = copy.deepcopy(entry)

            for key, value in entry_copy.items():
                if not (key.startswith("run_") and isinstance(value, dict)):
                    continue

                total_runs += 1
                trajectory = value.get("trajectory") or []
                if not isinstance(trajectory, list):
                    trajectory = []
                messages, error, removed = convert_trajectory_to_funccall_messages(
                    trajectory, args.add_in_context_learning_example
                )

                if messages is not None:
                    value["funccalloff_messages"] = messages
                    converted_runs += 1
                    value.pop("funccalloff_conversion_error", None)
                    if removed:
                        value["funccalloff_removed_duplicate_turns"] = removed
                    elif "funccalloff_removed_duplicate_turns" in value:
                        value.pop("funccalloff_removed_duplicate_turns")
                else:
                    value["funccalloff_messages"] = None
                    if error:
                        failure_reasons[error] = failure_reasons.get(error, 0) + 1
                        value["funccalloff_conversion_error"] = error
                    if "funccalloff_removed_duplicate_turns" in value:
                        value.pop("funccalloff_removed_duplicate_turns")

            json.dump(entry_copy, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(
        f"Wrote {total_instances} instances / {total_runs} runs to {args.save_file_path}."
        f" Successful conversions: {converted_runs}."
    )
    if failure_reasons:
        print("Conversion issues:")
        for reason, count in failure_reasons.items():
            print(f"  {reason}: {count}")


if __name__ == "__main__":
    main()
