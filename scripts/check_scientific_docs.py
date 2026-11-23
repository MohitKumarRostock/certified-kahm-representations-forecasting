#!/usr/bin/env python3
"""Check scientific Markdown rendering, internal links, and the results table.

Standard-library only. This is a documentation check, not a replacement for
numerical certificate verification or scientific review.
"""
from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote
import json
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
FENCE = re.compile(r"^\s*(?P<mark>~{3,}|" + re.escape(chr(96)) + r"{3,})")
MATH_DOLLAR = re.compile(r"(?<!\\)\$")
INLINE_MATH = re.compile(r"(?<!\\)\$([^$]*)\$")
LINK = re.compile(r"!?\[[^\]]+\]\(([^)\s]+)")
# GitHub rejects \operatorname and can misrender \tag; keep math portable.
GITHUB_UNSAFE_MATH = re.compile(r"\\(operatorname\*?|tag|hbox|rm)(?=[^A-Za-z]|$)")
NUMERIC_TOLERANCE = Fraction(51, 100_000_000_000)  # nine-place display rounding


def visible_lines(path: Path):
    """Yield Markdown outside fenced code, including display math."""
    fence_char = None
    fence_length = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        match = FENCE.match(line)
        if match:
            mark = match.group("mark")
            if fence_char is None:
                fence_char, fence_length = mark[0], len(mark)
            elif mark[0] == fence_char and len(mark) >= fence_length:
                fence_char, fence_length = None, 0
            continue
        if fence_char is None:
            yield line_number, line
    if fence_char is not None:
        raise ValueError(f"{path.relative_to(ROOT)}: unclosed fenced code block")


def check_markdown(path: Path) -> list[str]:
    errors = []
    in_display = False
    for number, line in visible_lines(path):
        origin = f"{path.relative_to(ROOT)}:{number}"
        if line.strip() == chr(36) * 2:
            in_display = not in_display
            continue
        math_regions = [line] if in_display else [
            match.group(1) for match in INLINE_MATH.finditer(line)
        ]
        for region in math_regions:
            for macro in GITHUB_UNSAFE_MATH.finditer(region):
                errors.append(
                    f"{origin}: GitHub-incompatible math macro \\{macro.group(1)}"
                )
        if in_display:
            continue
        if "$$" in line:
            errors.append(f"{origin}: put display-math delimiters on separate lines")
        if len(MATH_DOLLAR.findall(line)) % 2:
            errors.append(f"{origin}: unbalanced inline-math delimiter")
        if line.lstrip().startswith("|"):
            for match in INLINE_MATH.finditer(line):
                if re.search(r"(?<!\\)\|", match.group(1)):
                    errors.append(f"{origin}: escape Markdown table pipes inside inline math")
        for match in LINK.finditer(line):
            destination = match.group(1)
            if destination.startswith(("https://", "http://", "mailto:", "#", "data:")):
                continue
            target_name = unquote(destination.split("#", 1)[0].split("?", 1)[0])
            if not target_name:
                continue
            target = (path.parent / target_name).resolve()
            if not target.is_relative_to(ROOT) or not target.exists():
                errors.append(f"{origin}: missing local link target {destination}")
    if in_display:
        errors.append(f"{path.relative_to(ROOT)}: unclosed display-math block")
    return errors


def as_fraction(value: dict[str, str]) -> Fraction:
    return Fraction(int(value["numerator"]), int(value["denominator"]))


def check_results_table() -> list[str]:
    summary = json.loads(
        (ROOT / "reports/lower_certificates_v1/summary.json").read_text(encoding="utf-8")
    )
    by_id = {a["class_id"]: a for a in summary["assessments"] if a["n"] == 4096}
    classes = {
        "RAW": ("raw_k2_1", 2),
        "KAHM MEM": ("kahm_mem_k2_1", 2),
        "KAHM LOCAL": ("kahm_local_k2_1", 2),
        "Short history": ("short_history_k4_1", 4),
        "Full history": ("full_history_k4_1", 4),
    }
    pattern = re.compile(
        r"^\|\s*(RAW|KAHM MEM|KAHM LOCAL|Short history|Full history),"
        r"\s*\$\\kappa=(\d+)\$\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)"
        r"\s*\|\s*(Yes|No)\s*\|\s*(Requirement met|Class excluded)\s*\|"
    )
    seen = set()
    errors = []
    for number, line in enumerate((ROOT / "README.md").read_text(encoding="utf-8").splitlines(), 1):
        match = pattern.match(line)
        if match is None:
            continue
        name, kappa_text, lower, upper, admission, decision = match.groups()
        record_id, kappa = classes[name]
        seen.add(name)
        record = by_id[record_id]
        for column, displayed, exact in (
            ("lower", lower, record["lower"]),
            ("fitted upper", upper, record["fitted_upper"]),
        ):
            if exact is None or abs(Fraction(displayed) - as_fraction(exact)) > NUMERIC_TOLERANCE:
                errors.append(f"README.md:{number}: inconsistent {column} endpoint for {name}")
        expected_admission = "Yes" if record["admissible"] else "No"
        expected_decision = {
            "requirement_met": "Requirement met",
            "class_excluded": "Class excluded",
        }.get(record["status"])
        if (int(kappa_text), admission, decision) != (kappa, expected_admission, expected_decision):
            errors.append(f"README.md:{number}: inconsistent constraint or decision for {name}")
    if seen != set(classes):
        errors.append(f"README.md: result table is incomplete; found {sorted(seen)}")
    return errors


def main() -> int:
    errors = []
    for doc in DOCS:
        errors.extend(check_markdown(doc))
    errors.extend(check_results_table())
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(f"Scientific documentation check failed: {len(errors)} issue(s)", file=sys.stderr)
        return 1
    print(f"Scientific documentation check passed: {len(DOCS)} Markdown files and five recorded results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
