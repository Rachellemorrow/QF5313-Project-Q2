"""Render genuine OpenCode V2 JSON exports losslessly inside Markdown fences.

Run after the final assistant response for a complete final transcript.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", required=True)
    args = parser.parse_args()
    chunks = ["# Authentic OpenCode session export\n",
              "Native JSON exports are embedded without synthesizing dialogue. "
              "A live export ends at its export time; rerun after the final response.\n"]
    for session in args.sessions:
        if not session.startswith("ses_") or not session.replace("_", "").isalnum():
            raise ValueError("Invalid session identifier")
        # Installed CLI's `session export` truncated a large active transcript;
        # the documented native API returns complete validated JSON.
        raw = subprocess.check_output(["opencode", "api", "get",
            f"/api/experimental/session/{session}/export"], cwd=ROOT, text=True)
        payload = json.loads(raw)["data"]
        assert payload["info"]["id"] == session
        assert isinstance(payload["messages"], list) and payload["messages"]
        # Keep full native structure: tool arguments/results and every message type.
        fence = "`" * (max((len(part) for part in raw.split() if set(part) == {"`"}), default=3)+4)
        chunks.append(f"\n## {session}\n\nNative export SHA-256: `{hashlib.sha256(raw.encode()).hexdigest()}`\n\n{fence}json\n{raw}\n{fence}\n")
    (ROOT/"SESSION.md").write_text("\n".join(chunks))
    print(f"Exported {len(args.sessions)} genuine sessions to SESSION.md; review for secrets before commit.")


if __name__ == "__main__":
    main()
