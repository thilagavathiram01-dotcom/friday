import json
import re


def extract_first_json(text):
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def parse_tool_call(text):
    obj = extract_first_json(text)
    if not isinstance(obj, dict):
        return None
    if "name" not in obj:
        return None
    args = obj.get("arguments", obj.get("args", {}))
    if not isinstance(args, dict):
        try:
            args = json.loads(args) if isinstance(args, str) else {}
        except Exception:
            args = {}
    return {"name": obj["name"], "arguments": args}
