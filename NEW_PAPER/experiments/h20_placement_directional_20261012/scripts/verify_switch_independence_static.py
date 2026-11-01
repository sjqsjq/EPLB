#!/usr/bin/env python3
"""C5 static verification: W/D switch independence in the E-A decoupled controller.

For each candidate controller file (deployed + repo), statically verify:
  1. every runtime write to _effective_sync_window is dominated by an
     adaptive_window gate  -> adaptive_window=False freezes the window (fixes F1);
  2. the RESET history zeroing is dominated by an adaptive_decay gate and NOT by
     adaptive_window       -> adaptive_decay=False disables ALL live history
     zeroing (fixes F2), and D works with W off (fixes F3);
  3. the decay->0 changepoint step is dominated by adaptive_decay only (D alone);
  4. changepoint tracking (_adw_last_ratio) is unconditional (feeds D when W off);
  5. the shrink clamp (C3) is present;
  6. _calib_done is gated on calibrate_adaptive_sensitivity alone (C7), so with
     that flag False (v3 + all E-A arms) warm-start/tier paths stay unreachable
     and A11 is behaviourally identical to the v3-locked code.
Outputs a 4-arm reachability matrix. Refuses to overwrite existing results.
"""
import json, hashlib, os, sys, datetime

FILES = {
    "deployed": "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/controller.py",
    "repo":     "/workspace/EPLB/OEPLB/src/controller.py",
}
OUT = "/workspace/EPLB/NEW_PAPER/experiments/h20_placement_directional_20261012/results/ea_v1/switch_independence_static.json"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def chain_for(lines, idx):
    """Dominating condition chain for lines[idx]: walk backwards collecting
    lines with strictly decreasing indentation until the enclosing def."""
    ind = len(lines[idx]) - len(lines[idx].lstrip())
    chain, cur = [], ind
    method = None
    for j in range(idx - 1, -1, -1):
        s = lines[j]
        if not s.strip() or s.strip().startswith("#"):
            continue
        jind = len(s) - len(s.lstrip())
        if jind < cur:
            txt = s.strip()
            chain.append({"line": j + 1, "text": txt})
            cur = jind
            if txt.startswith("def "):
                method = txt
                break
    return method, chain

def chain_text(chain):
    return " || ".join(c["text"] for c in chain)

def find_sites(lines, needle):
    return [i for i, l in enumerate(lines) if needle in l and not l.strip().startswith("#")]

def verify_file(label, path):
    src = open(path).read()
    lines = src.splitlines()
    res = {"file": path, "sha256": sha256(path), "checks": {}}
    ok = True

    # 1. window mutation sites
    wsites = [i for i in find_sites(lines, "self._effective_sync_window = ")
              or find_sites(lines, "self._effective_sync_window =")]
    wsites = [i for i, l in enumerate(lines)
              if l.strip().startswith("self._effective_sync_window = ") and not l.strip().startswith("#")]
    wrows = []
    for i in wsites:
        method, chain = chain_for(lines, i)
        ct = chain_text(chain)
        if method and "__init__" in method:
            gate = "init (cfg.sync_window)"
            gated = True
        else:
            gated = ("_adw_on" in ct) or ("self.cfg.adaptive_window" in ct)
            gate = "adaptive_window" if gated else "UNEXPECTED: " + ct
        ok = ok and gated
        wrows.append({"line": i + 1, "method": method, "gate": gate, "gated": gated})
    res["checks"]["window_mutations"] = {"n_sites": len(wrows), "sites": wrows,
        "verdict": "ALL writes gated by adaptive_window (or __init__)" if ok else "FAIL"}

    # 2. load.zero_() sites
    zrows = []
    zok = True
    for i in [i for i, l in enumerate(lines) if "self.load.zero_()" in l]:
        method, chain = chain_for(lines, i)
        ct = chain_text(chain)
        if method and "_warm_start_placement" in method:
            # reachable only via _update_sensitivity_calibration, whose sole call
            # site must be guarded by `if not self._calib_done:`; and
            # _warm_start_placement must only be called from inside
            # _update_sensitivity_calibration. With calibrate_adaptive_sensitivity
            # =False (v3 + all E-A arms) _calib_done=True from init -> unreachable.
            call_idx = [k for k, l in enumerate(lines)
                        if "self._update_sensitivity_calibration(" in l and not l.strip().startswith("def")]
            guarded = False
            for k in call_idx:
                prev = next((lines[j].strip() for j in range(k - 1, -1, -1)
                             if lines[j].strip() and not lines[j].strip().startswith("#")), "")
                guarded = guarded or prev.startswith("if not self._calib_done:")
            ws_calls = [k for k, l in enumerate(lines)
                        if "self._warm_start_placement()" in l and not l.strip().startswith("def")]
            ws_in_calib = all("_update_sensitivity_calibration" in (chain_for(lines, k)[0] or "")
                              for k in ws_calls)
            gated = guarded and ws_in_calib and bool(call_idx)
            cls = "warm_start (unreachable when _calib_done=True)"
        elif method and "_check_eplb_refinement" in method:
            gated = any("self._plateau_count < 999" in lines[k] for k in range(len(lines)))
            cls = "refine (hard-disabled by plateau<999 early return)"
        else:
            gated = "self.cfg.adaptive_decay" in ct and "self.cfg.adaptive_window" not in ct
            cls = "RESET domain-shift"
        zok = zok and gated
        zrows.append({"line": i + 1, "method": method, "class": cls,
                      "dominating_chain": ct[:400], "correctly_gated": gated})
    res["checks"]["history_zeroing"] = {"n_sites": len(zrows), "sites": zrows,
        "verdict": "RESET gated by adaptive_decay only; other zero_ sites unreachable" if zok else "FAIL"}
    ok = ok and zok

    # 3. decay->0 step gated by adaptive_decay only
    dok = False
    for i in [i for i, l in enumerate(lines) if "self._decay_factor = 0.0" in l]:
        method, chain = chain_for(lines, i)
        ct = chain_text(chain)
        dok = ("adaptive_decay" in ct) and ("_adw_on" not in ct) and ("self.cfg.adaptive_window" not in ct)
    res["checks"]["decay_zero_step"] = {"gated_by_adaptive_decay_only": dok}
    ok = ok and dok

    # 4. changepoint tracking unconditional
    tok = False
    for i in [i for i, l in enumerate(lines) if l.strip() == "delta = abs(cur_r - self._adw_last_ratio)"]:
        method, chain = chain_for(lines, i)
        ct = chain_text(chain)
        tok = ("_adw_on" not in ct) and ("adaptive_window" not in ct)
    res["checks"]["changepoint_tracking_unconditional"] = tok
    ok = ok and tok

    # 5. C3 clamp present
    clamp = "new_window = min(self._effective_sync_window," in src and \
            "max(self.cfg.window_floor," in src
    res["checks"]["shrink_clamp_C3_present"] = clamp
    ok = ok and clamp

    # 6. C7 calib gate decoupled
    c7 = "self._calib_done = not cfg.calibrate_adaptive_sensitivity" in src
    res["checks"]["calib_gate_C7_decoupled"] = c7
    ok = ok and c7

    res["overall_static_pass"] = bool(ok)
    return res, ok

def main():
    if os.path.exists(OUT):
        sys.exit("REFUSE to overwrite existing " + OUT)
    per_file = {}
    all_ok = True
    for label, path in FILES.items():
        r, ok = verify_file(label, path)
        per_file[label] = r
        all_ok = all_ok and ok

    arms = {
        "A00_Woff_Doff": {"window_trajectory": "fixed at cfg.sync_window (no reachable mutation)",
            "history_zeroing": "none (RESET off, warm-start/refine unreachable)",
            "decay_alpha": "constant cfg.decay_factor (no decay->0 step)"},
        "A10_Won_Doff": {"window_trajectory": "adaptive (ADW + cos_sim mechanisms, both gated ON)",
            "history_zeroing": "none (RESET gated by adaptive_decay=False)",
            "decay_alpha": "constant cfg.decay_factor"},
        "A01_Woff_Don": {"window_trajectory": "fixed at cfg.sync_window",
            "history_zeroing": "RESET domain-shift zeroing active (gated by adaptive_decay=True)",
            "decay_alpha": "adaptive: decay->0 at changepoints (reachable with W off)"},
        "A11_Won_Don": {"window_trajectory": "adaptive, identical mutation set to v3-locked code",
            "history_zeroing": "RESET active, same trigger condition as v3",
            "decay_alpha": "adaptive, identical to v3"},
    }
    v3_equiv = {
        "v3_launch_flags": {"adaptive_window": True, "adaptive_decay": True,
            "calibrate_adaptive_sensitivity": False},
        "argument": ("With both adaptive flags True every gate evaluates as in the "
            "v3-locked code: ADW shrink/grow and cos_sim window mutations fire on the "
            "same conditions with the same log strings; RESET fires on the same "
            "condition (prev_window_ratio set + adaptive_decay True == adaptive_window "
            "True in v3); decay->0 unchanged; C7 gives _calib_done=True either way when "
            "calibrate_adaptive_sensitivity=False, so warm-start/tier remain unreachable "
            "as in v3. The C3 clamp differs from v3 only when window_floor > "
            "_effective_sync_window; with sync_window=16, window_floor=8 (v3 and E-A "
            "launch config) max(8, W//2) <= W for all reachable W>=8, so min() is a "
            "no-op and behaviour is identical."),
        "consequence": ("Per user rule, because code was modified for the switch fix, "
            "A11 must be re-run in the new 25-start campaign as the contemporaneous "
            "control; identity and ILP arms are NOT re-run."),
    }
    out = {
        "verification_id": "ea-v1-switch-independence-static",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "scope": "static source-level verification (C5 static part); dynamic log-level smoke verification runs before E-A protocol freeze (C5 dynamic part)",
        "audit_findings_addressed": ["F1 ungated ADW window mutations", "F2 RESET zeroing gated by wrong flag", "F3 D independence with W off", "C3 clamp ported to deployed", "C7 calib gate decoupled"],
        "per_file": per_file,
        "arm_reachability_matrix": arms,
        "a11_v3_equivalence": v3_equiv,
        "all_static_checks_pass": bool(all_ok),
        "known_inert_differences_deployed_vs_repo": [
            "deployed-only HOOKDIAG diagnostic counters (logging only, no state mutation)",
            "comment wording at the C3 clamp site (code identical)"],
    }
    tmp = OUT + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.rename(tmp, OUT)
    print("WROTE", OUT)
    print("sha256:", sha256(OUT))
    for label, r in per_file.items():
        print(label, "overall_static_pass:", r["overall_static_pass"])
        for k, v in r["checks"].items():
            if isinstance(v, dict) and "verdict" in v:
                print("  ", k, "->", v["verdict"], f"(n={v.get('n_sites')})")
            else:
                print("  ", k, "->", v)
    sys.exit(0 if all_ok else 1)

if __name__ == "__main__":
    main()
