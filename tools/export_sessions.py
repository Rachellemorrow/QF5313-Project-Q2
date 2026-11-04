"""Render genuine OpenCode V2 JSON exports losslessly inside Markdown fences.

Run after the final assistant response for a complete final transcript.
"""
import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", nargs="+", required=True)
    parser.add_argument("--final", action="store_true", help="Require idle/completed sessions and all referenced child IDs")
    parser.add_argument("--replace", action="store_true", help="Explicitly replace a previous generated export")
    args = parser.parse_args()
    if (ROOT/"SESSION.md").exists() and not args.replace:
        raise SystemExit("SESSION.md already exists; use --replace deliberately")
    manifest = {"exported_at_utc": datetime.now(timezone.utc).isoformat(), "final": args.final, "sessions": []}
    chunks = ["# Authentic OpenCode session export\n",
              "Native JSON exports are embedded without synthesizing dialogue. "
              "A live export ends at its export time; rerun after the final response.\n"]
    for session in args.sessions:
        if not session.startswith("ses_") or not session.replace("_", "").isalnum():
            raise ValueError("Invalid session identifier")
        # Installed CLI's `session export` truncated a large active transcript;
        # the documented native API returns complete validated JSON.
        directory = ROOT/".cache/session-exports"
        directory.mkdir(parents=True, exist_ok=True)
        native = directory/f"{session}.json"
        # A regular-file stdout also avoids the installed CLI's large-pipe
        # truncation; validate the complete file before writing SESSION.md.
        with native.open("w") as handle:
            subprocess.run(["opencode", "api", "get",
                f"/api/experimental/session/{session}/export"], cwd=ROOT,
                stdout=handle, check=True, text=True)
        raw = native.read_text()
        payload = json.loads(raw)["data"]
        if payload["info"]["id"] != session or not isinstance(payload["messages"], list) or not payload["messages"]:
            raise ValueError("Invalid native session export")
        if args.final and payload["messages"][-1]["type"] != "idle":
            raise ValueError(f"Session is not idle; final export must run after completion: {session}")
        if args.final and session == args.sessions[0]:
            referenced = set(re.findall(r"ses_[A-Za-z0-9]+", raw))
            if referenced-set(args.sessions):
                raise ValueError(f"Referenced sessions missing from manifest: {referenced-set(args.sessions)}")
        patterns = [r"sk-[A-Za-z0-9_-]{24,}", r"Bearer [A-Za-z0-9._-]{24,}"]
        redactions = 0
        for pattern in patterns:
            raw, count = re.subn(pattern, "[REDACTED_CREDENTIAL]", raw)
            redactions += count
        # Hash the exact embedded (possibly redacted) JSON, not the original.
        manifest["sessions"].append({"id": session, "parent_id": payload["info"].get("parentID"),
            "agent": payload["info"].get("agent"), "model": payload["info"].get("model"),
            "message_count": len(payload["messages"]), "first_message_id": payload["messages"][0]["id"],
            "last_message_id": payload["messages"][-1]["id"],
            "tool_parts": sum(part.get("type") == "tool" for message in payload["messages"] for part in message.get("content", [])),
            "redactions": redactions, "embedded_sha256": hashlib.sha256(raw.encode()).hexdigest()})
        # Keep full native structure: tool arguments/results and every message type.
        fence = "`" * (max((len(part) for part in raw.split() if set(part) == {"`"}), default=3)+4)
        chunks.append(f"\n## {session}\n\nNative export SHA-256: `{hashlib.sha256(raw.encode()).hexdigest()}`\n\n{fence}json\n{raw}\n{fence}\n")
    chunks.insert(1, "```json\n"+json.dumps(manifest, indent=2)+"\n```\n")
    (ROOT/"SESSION.md").write_text("\n".join(chunks))
    print(f"Exported {len(args.sessions)} genuine sessions to SESSION.md; review for secrets before commit.")


if __name__ == "__main__":
    main()
