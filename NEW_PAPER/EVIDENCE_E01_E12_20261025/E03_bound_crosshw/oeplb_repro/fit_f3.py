"""Fit the forced-imbalance sweep with (a) the paper's affine/Amdahl model
T(r) = A + B*r and (b) a hinge model T(r) = T0 + B*max(0, r - r_k), then report
the removable-stall fraction and the corrected throughput bound.

The hinge matters because balancing below the knee r_k buys nothing, so the
usable range is x_eff = (r_b - max(r_a, r_k))/r_b, not 1 - r_a/r_b.
"""
import json, glob, re, sys, statistics as st

RES = "/workspace/EPLB/OEPLB/benchmarks/results"
PLC = "/workspace/logs/" + (sys.argv[4] if len(sys.argv) > 4 else "plc57b.txt")
TAGPFX = sys.argv[5] if len(sys.argv) > 5 else "_d12_"
R_B = float(sys.argv[1]) if len(sys.argv) > 1 else 1.218   # natural / baseline r
R_A = float(sys.argv[2]) if len(sys.argv) > 2 else 1.04    # r the balancer holds

rmap = {}
for line in open(PLC):
    m = re.match(r"\s*(\S+)\s+r_avg=([\d.]+)", line)
    if m:
        rmap[m.group(1)] = float(m.group(2))
if "identity" in rmap:
    rmap["id"] = rmap.pop("identity")

HOLD = sys.argv[3].split(",") if len(sys.argv) > 3 else []

pts, rows, held = [], [], []
for tag, r in sorted(rmap.items(), key=lambda kv: kv[1]):
    ts = []
    for f in sorted(glob.glob(f"{RES}/*{TAGPFX}{tag}_r*.json")):
        d = json.load(open(f))
        assert d["errors"] == 0, (f, d["errors"])
        ts.append(d["total_time_s"])
    if not ts:
        continue
    mu = st.mean(ts)
    sd = st.stdev(ts) if len(ts) > 1 else 0.0
    rows.append((tag, r, ts, mu, sd))
    (held if tag in HOLD else pts).append((r, mu) if tag not in HOLD else (tag, r, mu))

print(f"{'tag':>6} {'r_avg':>7} {'runs (s)':>22} {'mean':>8} {'sd':>6} {'cv%':>6}")
for tag, r, ts, mu, sd in rows:
    print(f"{tag:>6} {r:7.3f} {' '.join('%.2f' % t for t in ts):>22} "
          f"{mu:8.2f} {sd:6.2f} {100*sd/mu:6.2f}")

if len(pts) < 4:
    print("\nnot enough points yet"); sys.exit(0)


def lsq(xs, ys):
    n = len(xs); sx = sum(xs); sy = sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    B = (n * sxy - sx * sy) / den
    A = (sy - B * sx) / n
    return A, B


def r2(xs, ys, pred):
    yb = sum(ys) / len(ys)
    ssr = sum((y - pred(x)) ** 2 for x, y in zip(xs, ys))
    sst = sum((y - yb) ** 2 for y in ys)
    return 1 - ssr / sst, ssr


xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
A, B = lsq(xs, ys)
r2a, ssra = r2(xs, ys, lambda x: A + B * x)
print(f"\n[affine]  T(r) = {A:.2f} + {B:.2f}*r        R2={r2a:.4f}  RSS={ssra:.3f}")

best = None
lo, hi = min(xs), max(xs)
for i in range(1, 2001):
    rk = lo + (hi - lo) * i / 2001.0
    hx = [max(0.0, x - rk) for x in xs]
    T0, Bh = lsq(hx, ys)
    if Bh <= 0:
        continue
    rr, ss = r2(hx, ys, lambda h: T0 + Bh * h)
    if best is None or ss < best[0]:
        best = (ss, rk, T0, Bh, rr)
ssrh, rk, T0, Bh, r2h = best
print(f"[hinge ]  T(r) = {T0:.2f} + {Bh:.2f}*max(0, r-{rk:.3f})  R2={r2h:.4f}  RSS={ssrh:.3f}")
print(f"          RSS ratio affine/hinge = {ssra/ssrh:.2f}x")


def bound(name, r_b, r_a, T_of, slope):
    Tb = T_of(r_b)
    s = max(0.0, (Tb - T_of(r_a))) / Tb           # removable stall fraction
    f_slope = slope * r_b / Tb                     # slope-implied sensitive frac
    print(f"  {name:>7}: T({r_b:.3f})={Tb:.2f}s  T({r_a:.3f})={T_of(r_a):.2f}s  "
          f"removable={100*s:5.2f}%  ->  throughput bound {100*s/(1-s):+6.2f}%  "
          f"(slope-implied f={f_slope:.3f})")


print(f"\nbound from r_b={R_B:.3f} to r_a={R_A:.3f}:")
bound("affine", R_B, R_A, lambda r: A + B * r, B)
bound("hinge", R_B, R_A, lambda r: T0 + Bh * max(0.0, r - rk), Bh)
for tag, r, mu in held:
    ph = T0 + Bh * max(0.0, r - rk)
    pa = A + B * r
    print(f"\nheld out {tag!r} (r={r:.3f}): measured {mu:.2f}s | "
          f"hinge predicts {ph:.2f}s ({100*(ph-mu)/mu:+.2f}%) | "
          f"affine predicts {pa:.2f}s ({100*(pa-mu)/mu:+.2f}%)")

print(f"  hinge x_eff = (r_b - max(r_a, r_k))/r_b = "
      f"{(R_B - max(R_A, rk))/R_B:.3f}   (naive x = {1 - R_A/R_B:.3f})")
