import json
import subprocess
import threading


class MCPClient:
    def __init__(self, command, args=None, env=None):
        self.proc = subprocess.Popen(
            [command] + (args or []),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )
        self._id = 0
        self._lock = threading.Lock()
        self._initialize()

    def _request(self, method, params=None):
        with self._lock:
            self._id += 1
            rid = self._id
            msg = {"jsonrpc": "2.0", "id": rid, "method": method}
            if params is not None:
                msg["params"] = params
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self.proc.stdout.readline()
                if not line:
                    raise RuntimeError("mcp closed")
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("id") == rid:
                    if "error" in obj:
                        raise RuntimeError(obj["error"])
                    return obj.get("result")

    def _initialize(self):
        self._request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "Friday", "version": "0.1.0"},
        })
        try:
            self._request("notifications/initialized")
        except Exception:
            pass

    def list_tools(self):
        res = self._request("tools/list") or {}
        return res.get("tools", [])

    def call_tool(self, name, arguments=None):
        return self._request("tools/call", {"name": name, "arguments": arguments or {}})

    def close(self):
        try:
            self.proc.terminate()
        except Exception:
            pass
