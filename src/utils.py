"""
Shared utilities: ID generation and file I/O helpers.
"""
import json
import os


def generate_id(fff: int, ccc: int, ppp: int, ss: int) -> str:
    """
    Generate a HVB corpus ID.

    Format: HVB_fff.ccc.ppp.ss
    """
    return f"HVB_{fff:03d}.{ccc:03d}.{ppp:03d}.{ss:02d}"


def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def write_lines(lines: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

