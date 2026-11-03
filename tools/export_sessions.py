"""Export genuine OpenCode session payloads and render losslessly in Markdown.

Run once at a documented cutoff; do not repeatedly export the export itself.
Native OpenCode V2 exports JSON, so SESSION.md embeds exact full JSON payloads
in fenced blocks rather than reconstructing prompts or missing tool calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--output", type=Path, default=Path("SESSION.md"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output.resolve()
    if not target.is_relative_to(root):
        raise ValueError("Output must remain in repository")
    sections = ["# Authentic OpenCode V2 session export\n\n"
                "Native JSON transcript exports embedded losslessly in Markdown. "
                "This is a point-in-time cutoff, not a claim to include subsequent "
                "export/commit/final-response events. No historical project sessions "
                "have been reconstructed.\n"]
    for session in args.sessions:
        result = subprocess.run(["opencode", "session", "export", session],
                                cwd=root, check=True, capture_output=True, text=True)
        payload = result.stdout
        parsed = json.loads(payload)
        if not parsed:
            raise ValueError("Empty export")
        # No keys have been used in this workflow. Fail closed if an obvious
        # credential appears; never silently redact mathematical/tool content.
        import re
        if re.search(r"(?:sk-[A-Za-z0-9_-]{24,}|Bearer [A-Za-z0-9_-]{24,})", payload):
            raise ValueError("Potential credential: inspect and explicitly redact before export")
        digest = hashlib.sha256(payload.encode()).hexdigest()
        fence = "`" * max(4, max((len(m.group()) + 1 for m in re.finditer(r"`+", payload)), default=4))
        sections.append(f"\n## {session}\n\nNative payload SHA-256: `{digest}`\n\n{fence}json\n{payload}\n{fence}\n")
    target.write_text("".join(sections))
    print(f"Wrote {target.relative_to(root)} ({target.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
