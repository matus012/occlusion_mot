"""PostToolUse hook (Write|Edit): ruff on the edited .py file + quick pytest.

Failures are printed to stderr with exit code 2, which Claude Code feeds back to the
model as a blocking error — closing the edit->test->fix loop automatically.
Silently no-ops for non-Python files, files outside src/tests/scripts, and while the
venv toolchain is not yet installed (scaffold window).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PYTEST_NO_TESTS_COLLECTED = 5


def run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 1, f"TIMEOUT after {timeout}s: {' '.join(cmd)}"
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0
    file_path = (payload.get("tool_input") or {}).get("file_path") or ""
    if not file_path.endswith(".py") or not VENV_PY.exists():
        return 0
    try:
        rel = Path(file_path).resolve().relative_to(ROOT)
    except ValueError:
        return 0
    top = rel.parts[0] if rel.parts else ""
    if top not in {"src", "tests", "scripts"}:
        return 0

    problems: list[str] = []

    code, out = run([str(VENV_PY), "-m", "ruff", "check", str(rel)], timeout=60)
    if "No module named" in out:  # toolchain still installing
        return 0
    if code != 0:
        problems.append("RUFF:\n" + out[-3000:])

    if top in {"src", "tests"} and (ROOT / "tests").exists():
        code, out = run(
            [str(VENV_PY), "-m", "pytest", "-x", "-q", "--timeout", "120"], timeout=420
        )
        if code not in (0, PYTEST_NO_TESTS_COLLECTED) and "No module named" not in out:
            problems.append("PYTEST:\n" + out[-4000:])

    if problems:
        print("\n\n".join(problems), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
