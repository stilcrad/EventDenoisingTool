"""Read-only structural checks. A pass is not publication authorization."""
import ast
import hashlib
import json
import pickletools
from pathlib import Path
import re
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
ABS_PATH = re.compile(r"\b[A-Za-z]:[\\/]|/Users/|/home/")
TOKEN = re.compile(r"(?:sk-[A-Za-z0-9_-]{24,}|ghp_[A-Za-z0-9]{24,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def main():
    failures = []
    files = [p for p in ROOT.rglob("*") if p.is_file() and not any(s in {"__pycache__", "outputs", ".git"} for s in p.relative_to(ROOT).parts)]
    python_count = 0
    for p in files:
        if p.suffix == ".py":
            python_count += 1
            try:
                ast.parse(p.read_text(encoding="utf-8-sig"))
            except SyntaxError as exc:
                failures.append({"file": str(p.relative_to(ROOT)), "issue": f"syntax at line {exc.lineno}"})
        if p.suffix in {".py", ".json", ".csv", ".md", ".txt", ".cff"}:
            content = p.read_text(encoding="utf-8-sig")
            for line in content.splitlines():
                # Public OS font fallback locations are not user-specific paths.
                if ABS_PATH.search(line) and "Windows\\Fonts\\" not in line and p.name != "check_release.py":
                    failures.append({"file": str(p.relative_to(ROOT)), "issue": "absolute user/machine path"})
                    break
            if TOKEN.search(content):
                failures.append({"file": str(p.relative_to(ROOT)), "issue": "credential-like text"})
    records = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))["models"]
    for rec in records:
        p = ROOT / rec["path"]
        if hashlib.sha256(p.read_bytes()).hexdigest() != rec["sha256"]:
            failures.append({"file": rec["path"], "issue": "checkpoint checksum"})
        with ZipFile(p) as z:
            for name in z.namelist():
                if name.endswith("data.pkl"):
                    for _, value, _ in pickletools.genops(z.read(name)):
                        if isinstance(value, str) and (ABS_PATH.search(value) or TOKEN.search(value)):
                            failures.append({"file": rec["path"], "issue": "path or credential-like checkpoint metadata"})
    print(json.dumps({"python_files": python_count, "checkpoints": len(records), "structural_failures": failures,
                      "publication_ready": False, "pending": "License, redistribution rights, actual data access, full independent reproduction and final security review"}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
