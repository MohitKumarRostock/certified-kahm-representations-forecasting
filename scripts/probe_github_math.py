#!/usr/bin/env python3
"""Inspect GitHub's server-rendered Markdown math, not unprocessed LaTeX."""
import html
import json
import os
import re
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PATHS = [
    "README.md",
    "docs/SCIENTIFIC_REFERENCE.md",
    "docs/LOWER_CERTIFICATE_PROOF.md",
    "docs/FITTED_RISK_GUARANTEE.md",
    "docs/ENCODER_DESIGN.md",
    "docs/READOUT_DESIGN.md",
    "docs/ADAPTING_TO_NEW_DATA.md",
]
api = "https://api.github.com/markdown"
headers = {
    "Accept": "text/html",
    "Content-Type": "application/json",
    "User-Agent": "JMLR-scientific-documentation-validation",
    "X-GitHub-Api-Version": "2022-11-28",
}
if os.environ.get("GITHUB_TOKEN"):
    headers["Authorization"] = "Bearer " + os.environ["GITHUB_TOKEN"]
for path in PATHS:
    text = (ROOT / path).read_text(encoding="utf-8")
    body = json.dumps({"text": text, "mode": "gfm", "context": os.environ.get("GITHUB_REPOSITORY", "MohitKumarRostock/certified-kahm-representations-forecasting")}).encode("utf-8")
    response = urlopen(Request(api, data=body, headers=headers, method="POST"), timeout=35)
    rendered = response.read().decode("utf-8")
    print("FILE",path,"HTML bytes",len(rendered),"math renderer count",rendered.count("math-renderer"),
          "math-inline count",rendered.count("math-inline"),"math-display count",rendered.count("math-display"))
    for regex in ["math-renderer","math-inline","math-display","Missing close brace","Extra \\\\left"]:
        match = re.search(regex, rendered, re.I)
        if match:
            print(" MATCH",regex,"CONTEXT",repr(rendered[max(0,match.start()-180):match.start()+500]))
    out = Path(os.environ.get("RUNNER_TEMP","/tmp")) / "rendered-math"
    out.mkdir(exist_ok=True)
    (out / (path.replace("/","_") + ".html")).write_text(rendered, encoding="utf-8")
