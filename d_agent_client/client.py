from __future__ import annotations

import argparse
import json
import sys
from multiprocessing.connection import Client
from typing import Any, Dict, List, Optional, Tuple, Union


class Text2SQLIPCClient:

    def __init__(
        self,
        address: Union[str, Tuple[str, int]],
        authkey: Optional[str] = None,
        timeout: float = 600.0,
    ) -> None:
        self.address = address
        self.authkey = None if authkey is None else authkey.encode("utf-8")
        self.timeout = timeout
        self.conn = Client(address=self.address, authkey=self.authkey)

        hello = self._recv()
        if not (isinstance(hello, dict) and hello.get("ok") and hello.get("ready")):
            raise RuntimeError(f"Server not ready: {hello}")
        self.model = hello.get("model")
        self.engines = hello.get("engines")

    @classmethod
    def unix(cls, path: str, authkey: Optional[str] = None, timeout: float = 600.0) -> "Text2SQLIPCClient":
        return cls(path, authkey=authkey, timeout=timeout)

    @classmethod
    def tcp(cls, host: str, port: int, authkey: Optional[str] = None, timeout: float = 600.0) -> "Text2SQLIPCClient":
        return cls((host, int(port)), authkey=authkey, timeout=timeout)

    def _send(self, msg: Dict[str, Any]) -> None:
        self.conn.send(msg)

    def _recv(self) -> Any:
        if not self.conn.poll(self.timeout):
            raise TimeoutError("Timed out waiting for server.")
        return self.conn.recv()

    def ping(self) -> bool:
        self._send({"cmd": "ping"})
        r = self._recv()
        return bool(isinstance(r, dict) and r.get("ok") and r.get("pong"))

    def query(self, question: str, engine: str = "auto") -> str:
        if question is None or not str(question).strip():
            raise ValueError("question must be a non-empty string")

        payload = {
            "cmd": "query",
            "question": str(question),
            "engine": engine,
        }
        self._send(payload)
        r = self._recv()
        if not isinstance(r, dict) or not r.get("ok"):
            raise RuntimeError(f"Server error: {r}")
        return r["result"]["text"]

    def batch_query(self, questions: List[str], engine: str = "auto") -> List[str]:
        jobs = [{"question": q, "engine": engine} for q in questions]
        self._send({"cmd": "batch_query", "jobs": jobs})
        r = self._recv()
        if not isinstance(r, dict) or not r.get("ok"):
            raise RuntimeError(f"Server error: {r}")

        out: List[str] = []
        for item in r.get("results", []):
            if item.get("ok"):
                out.append(item["result"]["text"])
            else:
                out.append(f"[ERROR] {item.get('error')}")
        return out

    def close(self) -> None:
        try:
            self._send({"cmd": "shutdown"})
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass


# ---------------- CLI ----------------

def _print(s: str) -> None:
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def smoke(cli: Text2SQLIPCClient) -> None:
    _print(f"Model: {cli.model}")
    _print(f"Engines: {cli.engines}")

    _print("\n[ping]")
    _print(str(cli.ping()))

    tests = [
        "Question: Find posts about bitcoin Cutoff_time: 2025-01-01",
    ]
    _print("\n[single query]")
    _print(cli.query(tests[0], engine="vector"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Client for Text2SQL fixed SQL functions and vector search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python client.py --tcp 0.0.0.0:61001 --authkey secret123 --vector --q "Question: Find posts about bitcoin Cutoff_time: 2025-01-01"

  python client.py --tcp 0.0.0.0:61001 --authkey secret123 --hybrid --q "Question: Find posts about bitcoin Month: 2024-02 Cutoff_time: 2024-03-01"

  # getpostcoreinfo(post_id, [timestamp])
  python client.py --tcp 0.0.0.0:61001 --authkey secret123 --sql --getpostcoreinfo 1l1y7rf 2025-10-01

  # getcommentcoreinfo(comment_id, [timestamp])
  python client.py --tcp 0.0.0.0:61001 --authkey secret123 --sql --getcommentcoreinfo ll7qwso 2025-10-01

  # getpostcommentslist(comment_id, [timestamp, up, down, max_comments])
  python client.py --tcp 0.0.0.0:61001 --authkey secret123 --sql --getpostcommentslist 1l4gv6p 2026-10-01 2 2 100

  # getauthorhistorylist(author, [timestamp | max_posts], [max_comments])
  python client.py --tcp 0.0.0.0:61001 --authkey secret123 --sql --getauthorhistorylist 68x8y9s1 2025-10-01 20 30
        """,
    )
    ap.add_argument("--unix", type=str, help="UNIX socket path, e.g. /tmp/text2sql.sock")
    ap.add_argument("--tcp", type=str, help="TCP host:port, e.g. 0.0.0.0:61001")
    ap.add_argument("--authkey", type=str, default=None, help="Authentication key")
    ap.add_argument("--sql", action="store_true", help="Force SQL engine (for fixed functions)")
    ap.add_argument("--vector", action="store_true", help="Force vector engine")
    ap.add_argument("--hybrid", action="store_true", help="Force hybrid engine")
    ap.add_argument("--q", type=str, help="Send a single raw question / function string")
    ap.add_argument("--smoke", action="store_true", help="Run a small connectivity test")

    # Fixed SQL function arguments
    ap.add_argument("--getpostcoreinfo", nargs="*", metavar="PARAMS", help="getpostcoreinfo(post_id, timestamp?)")
    ap.add_argument("--getcommentcoreinfo", nargs="*", metavar="PARAMS", help="getcommentcoreinfo(comment_id, timestamp?)")
    ap.add_argument(
        "--getpostcommentslist",
        nargs="*",
        metavar="PARAMS",
        help="getpostcommentslist(comment_id, timestamp?, up?, down?, max_comments?)",
    )
    ap.add_argument(
        "--getauthorhistorylist",
        nargs="*",
        metavar="PARAMS",
        help="getauthorhistorylist(author, timestamp?|max_posts, max_comments?)",
    )

    args = ap.parse_args()

    if not args.unix and not args.tcp:
        ap.error("One of --unix or --tcp is required.")

    if args.unix:
        cli = Text2SQLIPCClient.unix(args.unix, authkey=args.authkey)
    else:
        host, port = args.tcp.split(":")
        cli = Text2SQLIPCClient.tcp(host, int(port), authkey=args.authkey)

    try:
        if args.smoke:
            smoke(cli)

        elif args.getpostcoreinfo is not None:
            params_str = ", ".join([f'"{p}"' for p in args.getpostcoreinfo]) if args.getpostcoreinfo else ""
            query = f"getpostcoreinfo({params_str})"
            engine = "sql" if args.sql else "auto"
            print(cli.query(query, engine=engine))

        elif args.getcommentcoreinfo is not None:
            params_str = ", ".join([f'"{p}"' for p in args.getcommentcoreinfo]) if args.getcommentcoreinfo else ""
            query = f"getcommentcoreinfo({params_str})"
            engine = "sql" if args.sql else "auto"
            print(cli.query(query, engine=engine))

        elif args.getpostcommentslist is not None:
            formatted_params: List[str] = []
            for i, p in enumerate(args.getpostcommentslist or []):
                if i == 0:
                    formatted_params.append(f'"{p}"')
                elif p.isdigit():
                    formatted_params.append(p)
                else:
                    formatted_params.append(f'"{p}"')
            params_str = ", ".join(formatted_params)
            query = f"getpostcommentslist({params_str})"
            engine = "sql" if args.sql else "auto"
            print(cli.query(query, engine=engine))

        elif args.getauthorhistorylist is not None:
            formatted_params = []
            for i, p in enumerate(args.getauthorhistorylist or []):
                if i == 0:
                    formatted_params.append(f'"{p}"')
                elif p.isdigit():
                    formatted_params.append(p)
                else:
                    formatted_params.append(f'"{p}"')
            params_str = ", ".join(formatted_params)
            query = f"getauthorhistorylist({params_str})"
            engine = "sql" if args.sql else "auto"
            print(cli.query(query, engine=engine))

        elif args.q:

            engine = "hybrid" if args.hybrid else ("vector" if args.vector else ("sql" if args.sql else "auto"))
            print(cli.query(args.q, engine=engine))

        else:
            _print("Type 'exit' to quit.")
            _print("Available functions: getpostcoreinfo, getcommentcoreinfo, getpostcommentslist, getauthorhistorylist")
            while True:
                try:
                    q = input("> ").strip()
                except (KeyboardInterrupt, EOFError):
                    break
                if q.lower() in {"exit", "quit"} or not q:
                    break
                try:
                    engine = "hybrid" if args.hybrid else ("vector" if args.vector else ("sql" if args.sql else "auto"))
                    print(cli.query(q, engine=engine))
                except Exception as e:
                    print(f"[ERROR] {e}")
    finally:
        cli.close()


if __name__ == "__main__":
    main()