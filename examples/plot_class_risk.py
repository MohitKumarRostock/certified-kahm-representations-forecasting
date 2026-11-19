"""Export scientific figures from a retained class-risk assessment.

This script only reads saved results. It does not verify a run, fit a model,
draw observations, or compute confidence endpoints. Run verify-class-risk
first. Matplotlib is available through the package's optional tutorial extra.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path


COLORS = {"lower": "#225ea8", "upper": "#237a57", "fitted": "#b76516",
          "reference": "#394653", "tolerance": "#a5343b", "band": "#dbe4ec"}


def _read(path):
    def reject_constant(value):
        raise ValueError(f"non-finite JSON value: {value}")
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant)


def _number(record):
    if record is None:
        return math.nan
    if type(record) is not dict or set(record) != {"numerator", "denominator"}:
        raise ValueError("expected an exact fraction record")
    exact = Fraction(int(record["numerator"]), int(record["denominator"]))
    value = float(exact)
    if not math.isfinite(value):
        raise ValueError("plot values must be finite")
    return value


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _label(row):
    names = {"raw": "RAW", "kahm_mem": "KAHM MEM", "kahm_local": "KAHM LOCAL",
             "short_history": "Short history", "full_history": "Full history"}
    kappa = Fraction(int(row["kappa"]["numerator"]), int(row["kappa"]["denominator"]))
    return f"{names.get(row['representation'], row['representation'])}  (κ={kappa})"


def _decorate(ax, *, xlabel=None, ylabel=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y" if ylabel == "Population MSE" else "x", color="#e8edf2", linewidth=0.7)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def _save(fig, output, stem, files):
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        files.append({"path": path.name, "sha256": _hash(path)})


def _class_intervals(plt, rows, tolerance, n, repetition):
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10.8, max(4.6, 0.82 * len(rows) + 2.3)))
    visible = [tolerance]
    for y, row in enumerate(rows):
        lo, hi = _number(row.get("optimal_risk_lower")), _number(row.get("optimal_risk_upper"))
        lower, upper = _number(row.get("lower")), _number(row.get("class_upper"))
        fitted, fitted_upper = _number(row.get("fitted_risk")), _number(row.get("fitted_upper"))
        visible.extend(x for x in (lo, hi, lower, upper, fitted, fitted_upper) if math.isfinite(x))
        if math.isfinite(lo) and math.isfinite(hi):
            ax.plot([lo, hi], [y - 0.18] * 2, color=COLORS["reference"], linewidth=4,
                    solid_capstyle="round", zorder=3)
            ax.scatter([lo, hi], [y - 0.18] * 2, marker="|", s=80, linewidths=1.8,
                       color=COLORS["reference"], zorder=4)
            if lo == hi:
                ax.scatter([lo], [y - 0.18], s=30, marker="D", color=COLORS["reference"], zorder=4)
        if math.isfinite(lower):
            ax.scatter([lower], [y + 0.03], marker="|", s=190, linewidths=2.2,
                       color=COLORS["lower"], zorder=5)
        if math.isfinite(upper):
            ax.scatter([upper], [y + 0.03], marker="|", s=190, linewidths=2.2,
                       color=COLORS["upper"], zorder=5)
            if math.isfinite(lower):
                color = COLORS["band"] if lower <= upper else "#f1c8ca"
                ax.plot([lower, upper], [y + 0.03] * 2, linewidth=5, color=color, zorder=2)
        if math.isfinite(fitted):
            admissible = row["admissible"]
            ax.scatter([fitted], [y + 0.24], s=43,
                       marker="o" if admissible else "x",
                       color=COLORS["upper"] if admissible else COLORS["fitted"], zorder=5)
            if not admissible and math.isfinite(fitted_upper):
                ax.scatter([fitted_upper], [y + 0.24], s=45, marker=">",
                           facecolors="none", edgecolors=COLORS["fitted"], zorder=5)
        ax.text(1.01, y + 0.03, row["status"].replace("_", " "),
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=8.5)
    ax.axvline(tolerance, color=COLORS["tolerance"], linestyle=":", linewidth=1.4)
    ax.set_yticks(range(len(rows)), [_label(row) for row in rows])
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xlim(0, max(visible) * 1.12 if max(visible) > 0 else 1)
    _decorate(ax, xlabel="Population MSE")
    ax.set_title(f"Class-risk certificates and population references\n"
                 f"n = {n:,}; repetition {repetition}; τ = {tolerance:g}", loc="left", pad=15)
    handles = [
        Line2D([], [], color=COLORS["reference"], lw=4, label="Class optimum: verified reference interval"),
        Line2D([], [], color=COLORS["lower"], marker="|", markersize=12, linestyle="none", label="Class lower confidence endpoint"),
        Line2D([], [], color=COLORS["upper"], marker="|", markersize=12, linestyle="none", label="Class upper confidence endpoint"),
        Line2D([], [], color=COLORS["upper"], marker="o", linestyle="none", label="Fitted risk: admissible table"),
        Line2D([], [], color=COLORS["fitted"], marker="x", linestyle="none", label="Fitted risk: outside the class"),
        Line2D([], [], color=COLORS["fitted"], marker=">", markerfacecolor="none", linestyle="none", label="Upper endpoint for an inadmissible fitted table"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2,
              frameon=False, fontsize=8.5, columnspacing=1.7)
    fig.subplots_adjust(left=0.20, right=0.76, bottom=0.25, top=0.85)
    return fig


def _reference_scores(run, row, source_hashes):
    scores = {block["block_size"]: block.get("expected_block_score") for block in row["blocks"]}
    if all(value is not None for value in scores.values()):
        return scores
    # Earlier summaries may lack the convenience reference fields. Read only
    # the saved audit, never call the population oracle during plotting.
    path = (run / "audits" / f"{row['class_id']}.json").resolve()
    if not path.is_relative_to(run) or not path.is_file():
        return scores
    source_hashes[str(path.relative_to(run))] = _hash(path)
    reference = _read(path).get("analytic_partition", {}).get("hierarchy", [])
    saved = {item["block_size"]: item["expected_block_score"] for item in reference}
    return {b: value if value is not None else saved.get(b) for b, value in scores.items()}


def _hierarchy(plt, run, rows, tolerance, n, repetition, source_hashes):
    controls = [row for name in ("short_history", "full_history") for row in rows
                if row["representation"] == name]
    if not controls:
        return None
    fig, axes = plt.subplots(1, len(controls), figsize=(6.0 * len(controls), 4.6), squeeze=False)
    for ax, row in zip(axes[0], controls, strict=True):
        blocks = sorted(row["blocks"], key=lambda item: item["block_size"])
        bs = [block["block_size"] for block in blocks]
        reference = _reference_scores(run, row, source_hashes)
        expected = [_number(reference.get(b)) for b in bs]
        ax.plot(bs, expected, "D-", color=COLORS["reference"], markersize=4.5, linewidth=1.7,
                label=r"Population $\mathcal{W}_b$")
        ax.plot(bs, [_number(block["block_lower"]) for block in blocks], "o-",
                color=COLORS["lower"], markersize=4.5, label="Block lower endpoint")
        ax.plot(bs, [_number(block["empirical_lower"]) for block in blocks], "s--",
                color=COLORS["fitted"], markersize=4.5, label="Empirical lower, same witnesses")
        lo, hi = _number(row.get("optimal_risk_lower")), _number(row.get("optimal_risk_upper"))
        if math.isfinite(lo) and math.isfinite(hi):
            if lo == hi:
                ax.axhline(lo, color=COLORS["reference"], linestyle="--", linewidth=1,
                           label="Optimal class risk")
            else:
                ax.axhspan(lo, hi, color=COLORS["band"], label="Class-optimum reference interval")
        ax.axhline(tolerance, color=COLORS["tolerance"], linestyle=":", linewidth=1.3, label="Tolerance τ")
        ax.set_xscale("log", base=2)
        ax.set_xticks(bs, [str(b) for b in bs])
        ax.set_ylim(bottom=0)
        ax.set_title(_label(row), loc="left", fontsize=11)
        _decorate(ax, xlabel="Block size b", ylabel="Population MSE")
        if not any(math.isfinite(value) for value in expected):
            ax.text(0.03, 0.97, "Population hierarchy absent from saved records",
                    transform=ax.transAxes, va="top", fontsize=8, color=COLORS["reference"])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"Block hierarchy and finite-sample lower certificates\n"
                 f"n = {n:,}; repetition {repetition}", fontsize=13, x=0.07, ha="left")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.21, top=0.76, wspace=0.27)
    return fig


def _sample_sizes(plt, assessments, tolerance, repetition):
    rows = [row for row in assessments if row["repetition"] == repetition]
    if len({row["n"] for row in rows}) < 2:
        return None
    class_ids = list(dict.fromkeys(row["class_id"] for row in rows))
    columns = min(2, len(class_ids))
    count_rows = math.ceil(len(class_ids) / columns)
    fig, axes = plt.subplots(count_rows, columns, figsize=(6.1 * columns, 3.6 * count_rows), squeeze=False)
    for ax, class_id in zip(axes.flat, class_ids, strict=False):
        selected = sorted((row for row in rows if row["class_id"] == class_id), key=lambda row: row["n"])
        ns = [row["n"] for row in selected]
        ax.plot(ns, [_number(row["lower"]) for row in selected], "o-", color=COLORS["lower"],
                markersize=4.5, label="Combined class lower")
        class_uppers = [_number(row["class_upper"]) for row in selected]
        if any(math.isfinite(endpoint) for endpoint in class_uppers):
            ax.plot(ns, class_uppers, "s-", color=COLORS["upper"],
                    markersize=4.5, label="Admissible class upper")
        if not selected[0]["admissible"]:
            ax.plot(ns, [_number(row["fitted_upper"]) for row in selected], ">:",
                    color=COLORS["fitted"], markerfacecolor="none", markersize=4.5,
                    label="Inadmissible fitted upper")
        lo, hi = _number(selected[0].get("optimal_risk_lower")), _number(selected[0].get("optimal_risk_upper"))
        if math.isfinite(lo) and math.isfinite(hi):
            if lo == hi:
                ax.axhline(lo, color=COLORS["reference"], linestyle="--", linewidth=1,
                           label="Optimal class risk")
            else:
                ax.axhspan(lo, hi, facecolor=COLORS["band"], edgecolor=COLORS["reference"],
                           linewidth=0.8, label="Class-optimum reference interval")
        ax.axhline(tolerance, color=COLORS["tolerance"], linestyle=":", linewidth=1.2, label="Tolerance τ")
        ax.set_xscale("log", base=2)
        ax.set_xticks(ns, [f"{n:,}" for n in ns])
        ax.set_ylim(bottom=0)
        ax.set_title(_label(selected[0]), fontsize=11, loc="left")
        _decorate(ax, xlabel="Calibration episodes n", ylabel="Population MSE")
        ax.legend(loc="best", fontsize=7.5, frameon=False)
    for ax in list(axes.flat)[len(class_ids):]:
        ax.set_visible(False)
    fig.suptitle(f"Calibration-size comparison — repetition {repetition}\n"
                 "Shared prefixes; confidence allocation covers the declared study",
                 fontsize=13, ha="left", x=0.07)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def render_figures(run, output, *, sample_size=None, repetition=0):
    """Render saved values and return a JSON-serializable export manifest."""
    run = Path(run).resolve()
    output = Path(output).resolve()
    if output.exists() or output.is_relative_to(run):
        raise ValueError("figure output must be a new directory outside the retained run")
    summary_path, plan_path = run / "summary.json", run / "class_risk_plan.json"
    summary, plan = _read(summary_path), _read(plan_path)
    assessments = summary.get("assessments")
    if type(assessments) is not list or not assessments:
        raise ValueError("summary does not contain class-risk assessments")
    ns = sorted({row["n"] for row in assessments})
    sample_size = ns[-1] if sample_size is None else sample_size
    rows = [row for row in assessments if row["n"] == sample_size and row["repetition"] == repetition]
    if not rows:
        raise ValueError("requested sample size and repetition are absent from the run")
    tolerance = _number(plan["tolerance"])
    source_hashes = {"summary.json": _hash(summary_path), "class_risk_plan.json": _hash(plan_path)}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError("install the tutorial extra: python -m pip install -e '.[tutorial]'") from error
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "savefig.facecolor": "white"})
    output.mkdir(parents=True, exist_ok=False)
    files = []
    figures = [
        ("class_risk_intervals", _class_intervals(plt, rows, tolerance, sample_size, repetition)),
        ("block_hierarchy", _hierarchy(plt, run, rows, tolerance, sample_size, repetition, source_hashes)),
        ("sample_size_tradeoff", _sample_sizes(plt, assessments, tolerance, repetition)),
    ]
    for stem, figure in figures:
        if figure is not None:
            _save(figure, output, stem, files)
            plt.close(figure)
    manifest = {
        "schema": "kahm.class_risk_figure_export.v1",
        "run_id": summary.get("run_id"),
        "selected_sample_size": sample_size,
        "selected_repetition": repetition,
        "source_sha256": source_hashes,
        "figures": files,
        "scope": "rendered retained values; no numerical verification or new experiment",
        "decimal_coordinates": "descriptive approximations; exact JSON fractions govern decisions",
        "missing_endpoint_rule": "unavailable endpoints omitted, never replaced by zero",
    }
    (output / "figure_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export PNG/PDF figures from a retained class-risk run")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--repetition", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        result = render_figures(args.run, args.output, sample_size=args.sample_size, repetition=args.repetition)
    except (ValueError, KeyError, TypeError, OSError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
