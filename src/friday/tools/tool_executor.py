from .mcp_client import MCPClient


class ToolExecutor:
    def __init__(self):
        self.clients = {}

    def register(self, server_name, client):
        self.clients[server_name] = client

    def find_tool(self, tool_name):
        for sname, client in self.clients.items():
            for t in client.list_tools():
                if t.get("name") == tool_name:
                    return sname, t
        return None, None

    def execute(self, tool_name, arguments):
        sname, tool = self.find_tool(tool_name)
        if sname is None:
            return {"error": f"unknown tool {tool_name}"}
        client = self.clients[sname]
        return client.call_tool(tool_name, arguments)

    def all_tools(self):
        out = []
        for sname, client in self.clients.items():
            for t in client.list_tools():
                t = dict(t)
                t["_server"] = sname
                out.append(t)
        return out

    def close_all(self):
        for c in self.clients.values():
            c.close()
