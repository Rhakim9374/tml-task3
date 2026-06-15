"""Per-run eval *trajectory* from the training logs (peak vs final vs trend).

collect_sweep only scores each run's saved best checkpoint -- it shows peaks but
not whether a run was still climbing or had started to overfit. This parses the
val lines in runlogs/*.out to show, per config: the peak score (and when), the
final score, the recent slope, and a verdict (CLIMBING / PLATEAUED / DECLINING).

CLIMBING  -> undertrained; its checkpoint understates it -> give it more epochs.
DECLINING -> robust overfitting; it already peaked -> best checkpoint is captured.

Pure text parsing -- runs on the login node, no GPU.

    python -m scripts.analyze_trends --glob "runlogs/sweep_*.out"
"""

import argparse
import glob
import os
import re

EPOCH_RE = re.compile(r"^\[epoch\s+(\d+)\]")
VAL_RE = re.compile(r"^\s*val\[([\w-]+)\]:\s*clean=([\d.]+)\s+robust=([\d.]+)\s+score=([\d.]+)")


def parse(path):
    """Return {epoch: (best_score, clean, robust, variant)} keeping the best variant per epoch."""
    cur, per_epoch = None, {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = EPOCH_RE.match(line)
            if m:
                cur = int(m.group(1))
                continue
            m = VAL_RE.match(line)
            if m and cur is not None:
                variant, clean, robust, score = m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4))
                prev = per_epoch.get(cur)
                if prev is None or score > prev[0]:
                    per_epoch[cur] = (score, clean, robust, variant)
    return per_epoch


def config_name(path):
    base = os.path.basename(path)
    m = re.match(r"(.+?)\.\d+\.\d+\.out$", base)  # <tag>.<cluster>.<proc>.out
    return m.group(1) if m else base


def verdict(series):
    """series: sorted list of (epoch, score). Classify the late-stage trend."""
    epochs = [e for e, _ in series]
    scores = [s for _, s in series]
    peak = max(scores)
    peak_ep = epochs[scores.index(peak)]
    final, final_ep = scores[-1], epochs[-1]
    # Mean per-eval change over the last (up to) two evals -- less noisy than a
    # single step, less window-sensitive than an endpoint diff.
    deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    recent = sum(deltas[-2:]) / max(1, len(deltas[-2:])) if deltas else 0.0
    gap = peak - final
    if gap > 0.01:
        v = "DECLINING"
    elif recent > 0.005:
        v = "CLIMBING"
    else:
        v = "PLATEAUED"
    return peak, peak_ep, final, final_ep, recent, v


def main():
    p = argparse.ArgumentParser(description="Eval-score trajectory per run from logs")
    p.add_argument("--glob", default="runlogs/sweep_*.out")
    p.add_argument("--tail", type=int, default=4, help="how many recent (epoch:score) points to show")
    args = p.parse_args()

    # One row per config name; if a config was rerun, keep the most complete log.
    best_log = {}
    for path in glob.glob(args.glob):
        per = parse(path)
        if not per:
            continue
        name = config_name(path)
        if name not in best_log or len(per) > len(best_log[name]):
            best_log[name] = per
    if not best_log:
        raise SystemExit(f"no eval lines found in {args.glob!r} (jobs may not have hit the first eval yet)")

    rows = []
    for name, per in best_log.items():
        series = sorted((e, v[0]) for e, v in per.items())
        peak, peak_ep, final, final_ep, recent, v = verdict(series)
        tail = "  ".join(f"{e}:{s:.3f}" for e, s in series[-args.tail:])
        rows.append((name, peak, peak_ep, final, final_ep, recent, v, tail))

    rows.sort(key=lambda r: r[1], reverse=True)  # by peak
    print(f"{'config':<14}{'peak(ep)':>12}{'final(ep)':>12}{'Δrecent':>9}  {'verdict':<10} last evals (epoch:score)")
    for name, peak, pe, final, fe, recent, v, tail in rows:
        print(f"{name:<14}{f'{peak:.3f}({pe})':>12}{f'{final:.3f}({fe})':>12}{recent:>+9.3f}  {v:<10} {tail}")

    climbing = [r[0] for r in rows if r[6] == "CLIMBING"]
    if climbing:
        print(f"\nstill climbing at the end (consider more epochs): {', '.join(climbing)}")


if __name__ == "__main__":
    main()
