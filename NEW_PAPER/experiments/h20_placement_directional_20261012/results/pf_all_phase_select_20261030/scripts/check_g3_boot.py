#!/usr/bin/env python3
# check_g3_boot.py - pf_all_phase_select_20261030 boot checker (facts + gates).
# Usage: python3 check_g3_boot.py ART LABEL MODE    MODE in smoke|formal|readyfail
# Facts-first: every field is READ from real artifacts; formats pinned against
# the P0_r1 production boot (ARMFLAGS / auto dead_zone / Init lines, v2p1_rank0
# .jsonl window rows, sp_run_end.json atexit dump, segment_meta.jsonl offsets,
# bench_client_v2 LABEL.json). Missing artifacts recorded as missing, never
# fabricated. rc: smoke 0=PASS / 3=FAIL (queue STOPS, no single-arm patches);
# formal 0=complete / 1=incomplete (queue logs, continues); readyfail 1.
import json, os, re, sys

ART = os.path.abspath(sys.argv[1])
LABEL = sys.argv[2]
MODE = sys.argv[3]
G3 = os.path.dirname(os.path.dirname(ART))
MIN_RT_EXPECTED = 32  # config.py min_record_tokens default, extracted
                             # from frozen source at materialization time
MOVED_MB_PER_OP = 37.74      # 2 x 18.87 MB FP8 expert copy, ESTIMATE only

OUT = {"label": LABEL, "mode": MODE, "art": ART}
fails = []

def check(name, cond, detail=None):
    OUT.setdefault("checks", {})[name] = {"pass": bool(cond), "detail": detail}
    if not cond:
        fails.append(name)

def read_lines(p):
    if not os.path.exists(p):
        return None
    with open(p, "r", errors="replace") as f:
        return f.read().splitlines()

bm = {}
p = os.path.join(ART, "boot_meta.json")
if os.path.exists(p):
    try:
        bm = json.load(open(p))
    except Exception:
        bm = {}
ARM = str(bm.get("arm", ""))
OUT["boot_meta_arm"] = ARM
EXPECT_RAP = "1" if ARM == "G3_ALL" else "0"

# ---------- server.log ----------
lines = read_lines(os.path.join(ART, "server.log"))
sl = {}
if lines is None:
    sl["missing"] = True
else:
    sl["n_lines"] = len(lines)
    blob = "\n".join(lines)
    arm_lines = [l for l in lines if "[PB-OEPLB-ARMFLAGS]" in l]
    sl["armflags_count"] = len(arm_lines)
    kv = re.compile(r"(\w+)=(\S+)")
    parsed = []
    for l in arm_lines:
        body = l.split("[PB-OEPLB-ARMFLAGS]", 1)[1]
        parsed.append(dict(kv.findall(body)))
    uniq = sorted(set(json.dumps(x, sort_keys=True) for x in parsed))
    sl["armflags_unique_count"] = len(uniq)
    sl["armflags_first"] = parsed[0] if parsed else None
    sl["auto_deadzone_count"] = blob.count("auto dead_zone_ratio")
    m = re.search(r"auto dead_zone_ratio = ([0-9.]+)", blob)
    sl["auto_deadzone_value"] = m.group(1) if m else None
    sl["init_count"] = blob.count("[PB-OEPLB] Init v")
    m2 = re.search(r"min_tok=(\d+)", blob)
    sl["init_min_tok"] = int(m2.group(1)) if m2 else None
    m3 = re.search(r"thresh=([0-9.]+)", blob)
    sl["init_thresh"] = m3.group(1) if m3 else None
    sl["controller_enabled_count"] = blob.count("controller enabled")
    sl["planhash_count"] = blob.count("[PB-OEPLB-PLANHASH]")
    sl["diag_count"] = blob.count("[PB-OEPLB-DIAG]")
    issued = re.findall(r"window#\d+: issued (\d+) swap", blob)
    sl["issued_line_count"] = len(issued)
    sl["issued_ops_total"] = sum(int(x) for x in issued)
    sl["moved_mb_estimate"] = round(sl["issued_ops_total"] * MOVED_MB_PER_OP, 1)
    sl["calib_count"] = blob.count("CALIB")
    sl["traceback_count"] = blob.count("Traceback")
    sl["cuda_oom_count"] = blob.count("CUDA out of memory")
    sl["ready_to_roll_count"] = blob.count("ready to roll")
    sl["init_by_eplb_count"] = blob.count("init_expert_location from init_by_eplb")
OUT["server_log"] = sl

# ---------- v2p1 + segments ----------
def agg_v2p1(rows):
    a = {"n_windows": len(rows)}
    if not rows:
        return a
    def s(key):
        t = 0
        for r in rows:
            try:
                t += int(r.get(key, 0) or 0)
            except Exception:
                pass
        return t
    for k in ("sp_valid_rows", "sp_recorded_steps", "n_fresh_assignments",
              "sp_sample_opportunities", "sp_cover_steps", "sp_swaps_started"):
        a["sum_" + k] = s(k)
    a["count_committed_true"] = sum(1 for r in rows if bool(r.get("committed")))
    a["count_event_nonempty"] = sum(1 for r in rows if r.get("event"))
    cw = []
    for r in rows:
        try:
            cw.append(float(r.get("c_window_ms", 0) or 0))
        except Exception:
            pass
    if cw:
        a["c_window_ms_mean"] = round(sum(cw) / len(cw), 3)
        a["c_window_ms_max"] = max(cw)
    pv = []
    for r in rows:
        try:
            pv.append(int(r.get("p2l_version", 0) or 0))
        except Exception:
            pass
    a["p2l_version_max"] = max(pv) if pv else None
    a["window_id_first"] = rows[0].get("window_id")
    a["window_id_last"] = rows[-1].get("window_id")
    a["desync_true_count"] = sum(1 for r in rows if bool(r.get("desync")))
    a["lockstep_false_count"] = sum(1 for r in rows if r.get("lockstep_ok") is False)
    rec = []
    for i, r in enumerate(rows):
        try:
            if int(r.get("sp_recorded_steps", 0) or 0) > 0:
                rec.append(i)
        except Exception:
            pass
    n = len(rows)
    a["recorded_window_first_idx"] = rec[0] if rec else None
    a["recorded_window_last_idx"] = rec[-1] if rec else None
    a["recorded_in_first_half"] = any(i < n // 2 for i in rec)
    a["recorded_in_second_half"] = any(i >= n // 2 for i in rec)
    return a

v2p1_all = []
vp = read_lines(os.path.join(ART, "v2p1_rank0.jsonl"))
if vp:
    for l in vp:
        l = l.strip()
        if not l:
            continue
        try:
            v2p1_all.append(json.loads(l))
        except Exception:
            v2p1_all.append({})
OUT["v2p1_total_rows"] = len(v2p1_all)

def bench_facts(seg_label):
    p = os.path.join(ART, str(seg_label) + ".json")
    if not os.path.exists(p):
        return {"missing": True}
    try:
        j = json.load(open(p))
    except Exception as e:
        return {"unreadable": str(e)}
    out = {}
    for k, v in j.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    out["total_time_s"] = j.get("total_time_s")
    return out

segs = []
smeta = read_lines(os.path.join(ART, "segment_meta.jsonl"))
if smeta:
    for l in smeta:
        l = l.strip()
        if not l:
            continue
        try:
            m = json.loads(l)
        except Exception:
            segs.append({"segment_meta_unparsable": l[:200]})
            continue
        b = int(m.get("v2p1_before", 0) or 0)
        a2 = int(m.get("v2p1_after", 0) or 0)
        segs.append({"meta": m,
                     "ds_base": os.path.basename(str(m.get("ds", ""))),
                     "v2p1": agg_v2p1(v2p1_all[b:a2]),
                     "bench": bench_facts(m.get("seg"))})
OUT["segments"] = segs
OUT["segment_count"] = len(segs)

# ---------- sp_run_end ----------
sre_p = os.path.join(ART, "sp_run_end.json")
if os.path.exists(sre_p):
    try:
        sre = json.load(open(sre_p))
        flat = {}
        for k, v in sre.items():
            flat[k + "_len" if isinstance(v, list) else k] = len(v) if isinstance(v, list) else v
        OUT["sp_run_end"] = flat
    except Exception as e:
        OUT["sp_run_end"] = {"unreadable": str(e)}
else:
    OUT["sp_run_end"] = {"missing": True}

# ---------- static launcher case-block diff (P5a) ----------
def static_case_diff():
    src = read_lines(os.path.join(G3, "scripts", "launch_arm_g3.sh"))
    if src is None:
        return {"missing": True}
    def block(name):
        i = None
        for k, l in enumerate(src):
            if l.strip() == name + ")":
                i = k
                break
        if i is None:
            return None
        j = None
        for k in range(i + 1, len(src)):
            if src[k].rstrip().endswith(";;"):
                j = k
                break
        if j is None:
            return None
        return [l.strip() for l in src[i + 1:j + 1]]
    pf = block("G3_PF")
    alb = block("G3_ALL")
    if pf is None or alb is None:
        return {"blocks_found": False}
    rap = "export OEPLB_RECORD_ALL_PHASE="
    pf_core = [l for l in pf if not l.startswith(rap)]
    al_core = [l for l in alb if not l.startswith(rap)]
    pf_r = [l for l in pf if l.startswith(rap)]
    al_r = [l for l in alb if l.startswith(rap)]
    return {"blocks_found": True, "cores_identical": pf_core == al_core,
            "rap_pf": pf_r, "rap_all": al_r,
            "rap_one_each_0_vs_1": pf_r == [rap + "0"] and al_r == [rap + "1"]}
OUT["static_case_diff"] = static_case_diff()

# ---------- min_record_tokens config evidence (P5c) ----------
CFG = "/opt/conda/lib/python3.11/site-packages/sglang/srt/managers/pb_oeplb/config.py"
if os.path.exists(CFG):
    OUT["cfg_min_record_tokens_lines"] = [l.strip() for l in open(CFG, errors="replace").read().splitlines()
                                          if "min_record_tokens" in l][:6]
else:
    OUT["cfg_min_record_tokens_lines"] = None

# ---------- checks ----------
def common_checks():
    check("armflags_count_8", sl.get("armflags_count") == 8)
    check("armflags_unique", sl.get("armflags_unique_count") == 1)
    af = sl.get("armflags_first") or {}
    check("armflags_rap_expected", af.get("record_all_phase") == EXPECT_RAP,
          {"got": af.get("record_all_phase"), "expect": EXPECT_RAP})
    check("armflags_sp_mode_fixed", af.get("sp_mode") == "fixed", af.get("sp_mode"))
    check("armflags_sp_s1_w16_rev0", af.get("sp_s") == "1" and af.get("sp_w") == "16"
          and af.get("sp_rev") == "0", af)
    check("deadzone_8_hits_1096", sl.get("auto_deadzone_count") == 8
          and str(sl.get("auto_deadzone_value", "")).startswith("1.096"))
    check("ready_to_roll_1", sl.get("ready_to_roll_count") == 1)
    check("traceback_0", sl.get("traceback_count") == 0)
    check("oom_0", sl.get("cuda_oom_count") == 0)
    check("calib_0", sl.get("calib_count") == 0)
    check("identity_layout_no_init_by_eplb", sl.get("init_by_eplb_count") == 0)
    sre = OUT["sp_run_end"]
    check("sp_run_end_present", "missing" not in sre and "unreadable" not in sre)
    if "missing" not in sre and "unreadable" not in sre:
        check("sre_mode_fixed", sre.get("mode") == "fixed", sre.get("mode"))
        check("sre_S1_H16_W16", sre.get("S_final") == 1 and sre.get("H_final") == 16
              and sre.get("W_final") == 16,
              {"S": sre.get("S_final"), "H": sre.get("H_final"), "W": sre.get("W_final")})

def seg_checks(expect_n):
    check("segment_count", OUT["segment_count"] == expect_n, OUT["segment_count"])
    for seg in OUT["segments"]:
        m = seg.get("meta", {})
        tag = str(m.get("seg", "seg"))
        check("bench_rc0_" + tag, m.get("bench_rc") == 0, m.get("bench_rc"))
        check("bench_json_" + tag, not seg.get("bench", {}).get("missing", False))
        check("v2p1_windows_" + tag, seg.get("v2p1", {}).get("n_windows", 0) > 0)

if MODE == "readyfail":
    OUT["note"] = "facts-only dump after READY_FAIL"
elif MODE == "formal":
    common_checks()
    seg_checks(1)
    for seg in OUT["segments"]:
        tag = str(seg.get("meta", {}).get("seg", "seg"))
        tt = (seg.get("bench") or {}).get("total_time_s")
        check("total_time_s_present_" + tag, isinstance(tt, (int, float)) and tt > 0, tt)
elif MODE == "smoke":
    common_checks()
    seg_checks(2)
    sre = OUT["sp_run_end"]
    drb = sre.get("decode_record_batches")
    if ARM == "G3_PF":
        check("P2_decode_record_batches_0", drb == 0, drb)
    if ARM == "G3_ALL":
        check("P1_decode_record_batches_gt0", isinstance(drb, int) and drb > 0, drb)
        pf_p = os.path.join(G3, "boots", "SM_PF", "g3_facts.json")
        pf_ok = os.path.exists(pf_p)
        check("P3_P4_P5_pf_facts_present", pf_ok)
        pf = {}
        if pf_ok:
            try:
                pf = json.load(open(pf_p))
            except Exception:
                pf = {}
        pf_segs = {}
        for s0 in pf.get("segments", []):
            pf_segs[s0.get("ds_base")] = s0
        for seg in OUT["segments"]:
            dsb = seg.get("ds_base")
            tag = str(seg.get("meta", {}).get("seg", "seg"))
            v = seg.get("v2p1", {})
            pfs = (pf_segs.get(dsb) or {}).get("v2p1", {})
            av = v.get("sum_sp_valid_rows")
            pv = pfs.get("sum_sp_valid_rows")
            check("P3_valid_rows_" + tag,
                  isinstance(av, int) and isinstance(pv, int) and av > pv,
                  {"all": av, "pf": pv})
            ar = v.get("sum_sp_recorded_steps")
            pr = pfs.get("sum_sp_recorded_steps")
            check("P3_recorded_steps_" + tag,
                  isinstance(ar, int) and isinstance(pr, int) and ar > pr,
                  {"all": ar, "pf": pr})
            check("P4_full_interval_" + tag,
                  bool(v.get("recorded_in_first_half")) and bool(v.get("recorded_in_second_half")),
                  {"first": v.get("recorded_in_first_half"),
                   "second": v.get("recorded_in_second_half")})
        scd = OUT["static_case_diff"]
        check("P5_static_single_delta",
              bool(scd.get("blocks_found")) and bool(scd.get("cores_identical"))
              and bool(scd.get("rap_one_each_0_vs_1")), scd)
        pf_sl = pf.get("server_log") or {}
        pf_af = pf_sl.get("armflags_first") or {}
        af = sl.get("armflags_first") or {}
        if pf_af and af:
            dk = sorted(k for k in set(list(pf_af.keys()) + list(af.keys()))
                        if pf_af.get(k) != af.get(k))
            check("P5_armflags_runtime_single_delta",
                  dk == ["record_all_phase"] and pf_af.get("record_all_phase") == "0"
                  and af.get("record_all_phase") == "1", {"diff_keys": dk})
        else:
            check("P5_armflags_runtime_single_delta", False, "armflags missing one side")
        check("P5_calib_zero_both", pf_sl.get("calib_count") == 0 and sl.get("calib_count") == 0)
        check("P5_init_min_tok_equal",
              sl.get("init_min_tok") is not None
              and pf_sl.get("init_min_tok") == sl.get("init_min_tok"),
              {"pf": pf_sl.get("init_min_tok"), "all": sl.get("init_min_tok")})
        check("P5_min_record_tokens_expected",
              MIN_RT_EXPECTED is not None
              and isinstance(OUT.get("cfg_min_record_tokens_lines"), list)
              and len(OUT["cfg_min_record_tokens_lines"]) > 0,
              {"expected_default": MIN_RT_EXPECTED})
else:
    print("FATAL unknown mode " + MODE)
    sys.exit(9)

all_pass = len(fails) == 0
OUT["all_pass"] = all_pass
OUT["fail_names"] = list(fails)
OUT["min_record_tokens_expected"] = MIN_RT_EXPECTED
with open(os.path.join(ART, "g3_facts.json"), "w") as f:
    json.dump(OUT, f, indent=1, sort_keys=True)
if MODE == "smoke" and ARM == "G3_ALL":
    with open(os.path.join(ART, "g3_smoke_gate.json"), "w") as f:
        json.dump({"all_pass": all_pass, "fails": list(fails),
                   "checks": OUT.get("checks", {})}, f, indent=1, sort_keys=True)
print("CHECK_G3 label=%s mode=%s arm=%s all_pass=%s fails=%s"
      % (LABEL, MODE, ARM, all_pass, ",".join(fails)))
if MODE == "readyfail":
    sys.exit(1)
sys.exit(0 if all_pass else (3 if MODE == "smoke" else 1))
