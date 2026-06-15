"""Per-run eval *trajectories* from the training logs -- BOTH robust scores.

collect_sweep/rank_robust score saved checkpoints; this shows the trend over
epochs, parsed from the val lines in runlogs/*.out. Each run logs a PGD-20 score
every eval and a strong CE+DLR score every --strong-eval-every epochs, so per
config we surface BOTH: the PGD-score peak/final (the leaderboard-proxy attack)
AND the strong-score peak/final (the AutoAttack proxy), their gap at the last
joint eval (large +gap = gradient masking), and the worst-case (min) score --
so we can read the combined picture, not just one attack.

CLIMBING (PGD)  -> undertrained; give it more epochs.
DECLINING (PGD) -> robust overfitting; its best checkpoint already captured it.

Pure text parsing -- runs on the login node, no GPU.

    python -m scripts.analyze_trends --glob "runlogs/sweep_*.out"
"""

import argparse
import glob
import os
import re

EPOCH_RE = re.compile(r"^\[epoch\s+(\d+)\]")
VAL_RE = re.compile(
    r"^\s*val\[([\w-]+)\]:\s*clean=([\d.]+)\s+robust=([\d.]+)\s+score=([\d.]+)"
    r"(?:\s+strong_robust=([\d.]+)\s+strong_score=([\d.]+))?")


def parse(path):
    """Return {epoch: {'pgd': best_pgd_score, 'strong': best_strong_score|None}}.

    Both scores are taken as the max over that epoch's variant lines (live/ema or
    dbn-clean/dbn-adv), mirroring the independently-saved .pt / .strong.pt picks.
    """
    cur, per = None, {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = EPOCH_RE.match(line)
            if m:
                cur = int(m.group(1))
                continue
            m = VAL_RE.match(line)
            if m and cur is not None:
                pgd = float(m.group(4))
                strong = float(m.group(6)) if m.group(6) else None
                e = per.setdefault(cur, {"pgd": None, "strong": None})
                if e["pgd"] is None or pgd > e["pgd"]:
                    e["pgd"] = pgd
                if strong is not None and (e["strong"] is None or strong > e["strong"]):
                    e["strong"] = strong
    return per


def config_name(path):
    base = os.path.basename(path)
    m = re.match(r"(.+?)\.\d+\.\d+\.out$", base)  # <tag>.<cluster>.<proc>.out
    return m.group(1) if m else base


def peak_final(series):
    """series: sorted [(epoch, score)]. -> (peak, peak_ep, final, final_ep)."""
    epochs = [e for e, _ in series]
    scores = [s for _, s in series]
    peak = max(scores)
    return peak, epochs[scores.index(peak)], scores[-1], epochs[-1]


def verdict(series):
    """Classify the late PGD-score trend: CLIMBING / PLATEAUED / DECLINING."""
    scores = [s for _, s in series]
    peak, final = max(scores), scores[-1]
    deltas = [scores[i] - scores[i - 1] for i in range(1, len(scores))]
    recent = sum(deltas[-2:]) / max(1, len(deltas[-2:])) if deltas else 0.0
    if peak - final > 0.01:
        return "DECLINING", recent
    if recent > 0.005:
        return "CLIMBING", recent
    return "PLATEAUED", recent


def main():
    p = argparse.ArgumentParser(description="Both-attack eval-score trajectories per run from logs")
    p.add_argument("--glob", default="runlogs/sweep_*.out")
    p.add_argument("--tail", type=int, default=4, help="how many recent PGD (epoch:score) points to show")
    args = p.parse_args()

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
        pgd_series = sorted((e, v["pgd"]) for e, v in per.items() if v["pgd"] is not None)
        strong_series = sorted((e, v["strong"]) for e, v in per.items() if v["strong"] is not None)
        p_peak, p_pe, p_fin, _ = peak_final(pgd_series)
        v, recent = verdict(pgd_series)
        if strong_series:
            s_peak, s_pe, _, le = peak_final(strong_series)
            s_fin = per[le]["strong"]
            gap = per[le]["pgd"] - s_fin   # PGD vs strong at the last joint eval
            worst = min(per[le]["pgd"], s_fin)
        else:
            s_peak = s_pe = s_fin = gap = None
            worst = p_fin
        tail = "  ".join(f"{e}:{s:.3f}" for e, s in pgd_series[-args.tail:])
        rows.append((name, p_peak, p_pe, p_fin, s_peak, s_pe, s_fin, gap, worst, v, tail))

    rows.sort(key=lambda r: r[8], reverse=True)  # by worst-case (min) score
    hdr = (f"{'config':<22}{'pgdPk(ep)':>12}{'pgdFin':>8}{'strPk(ep)':>12}{'strFin':>8}"
           f"{'gap':>8}{'min':>8}  {'verdict':<10} recent pgd (ep:score)")
    print(hdr)
    for name, p_peak, p_pe, p_fin, s_peak, s_pe, s_fin, gap, worst, v, tail in rows:
        spk = f"{s_peak:.3f}({s_pe})" if s_peak is not None else "-"
        sfn = f"{s_fin:.3f}" if s_fin is not None else "-"
        gp = f"{gap:+.3f}" if gap is not None else "-"
        mask = "  MASKED?" if (gap is not None and gap > 0.04) else ""
        print(f"{name:<22}{f'{p_peak:.3f}({p_pe})':>12}{p_fin:>8.3f}{spk:>12}{sfn:>8}"
              f"{gp:>8}{worst:>8.3f}  {v:<10} {tail}{mask}")

    climbing = [r[0] for r in rows if r[9] == "CLIMBING"]
    if climbing:
        print(f"\nstill climbing at the end (consider more epochs): {', '.join(climbing)}")
    print("ranked by min(pgd_score, strong_score) -- the worst case across attacks. "
          "gap = pgd_score - strong_score at the last joint eval; large +gap = masking.")


if __name__ == "__main__":
    main()
