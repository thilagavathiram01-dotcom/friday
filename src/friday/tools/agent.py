import json
from .tool_call_parser import parse_tool_call


def format_tools_for_prompt(tools):
    lines = []
    for t in tools:
        lines.append(json.dumps({"name": t.get("name"), "description": t.get("description", ""), "parameters": t.get("inputSchema", {})}))
    return "\n".join(lines)


class FridayAgent:
    def __init__(self, model_generate, executor, system_prompt=None):
        self.generate = model_generate
        self.executor = executor
        self.system_prompt = system_prompt or "You are Friday, a fast helpful assistant. Use tools when needed."

    def run(self, user_input, max_turns=4):
        tools = self.executor.all_tools()
        tool_block = format_tools_for_prompt(tools)
        history = [
            {"role": "system", "content": self.system_prompt + "\n\nAvailable tools:\n" + tool_block},
            {"role": "user", "content": user_input},
        ]
        for _ in range(max_turns):
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in history)
            out = self.generate(prompt)
            call = parse_tool_call(out)
            if call is None:
                return out
            result = self.executor.execute(call["name"], call["arguments"])
            history.append({"role": "assistant", "content": out})
            history.append({"role": "tool", "content": json.dumps(result)})
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        return self.generate(prompt)
