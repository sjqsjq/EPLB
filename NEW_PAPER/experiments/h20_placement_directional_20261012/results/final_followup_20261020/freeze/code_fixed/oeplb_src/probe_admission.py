"""Route-B bounded probe admission -- pure state machine (spec v1.1-calib).

Frozen protocol: PRE_REG_calibration_m1m2m3.md +
DRAFT_window_admission_method_v1_1.md (adw_v2p1/next_version).

Design contract (see PRE_REG sections A/C/F):
  * PURE python -- no torch, no I/O, no logging. The controller drives this
    machine and is responsible for all collectives and persistence.
  * propose() is called in phase 2 (BEFORE decision_hash): it never mutates
    segment/state data; it only computes the recommended override and
    stashes it as _tentative. advance() is called in phase 4 with the
    REDUCED (cross-rank MAX/MIN-verified) window series, and is the only
    mutator -- therefore every rank runs an identical state sequence
    (deterministic function of the reduced series).
  * License: AT MOST ONE probe event per run ("至多一次"; zero is a legal
    outcome if the trigger condition is never met). No retry after abort /
    contamination / DONE.
  * Growth gating (modes gate/aaa): ANY rec with W_after > W_before is
    reverted to W_before with event grow_gated_probe -- whole-run scope
    (ARMED, segments, DONE, ABORTED alike), per PRE_REG table A row
    "增长门控范围". Detection / shrink / consistency paths are NEVER
    overridden: S3-family events during a segment mean detector takeover
    -> ABORTED (contaminated), segments truncated but retained.
  * AAA arm: identical machine, no W override anywhere (全程 W16); the CAND
    segment runs n_cand*w_cand/w_base windows (6) so that per-segment
    forward budgets (64/96/64 steps) correspond exactly to ABA.
  * eps_risk is an ABORT THRESHOLD, not a proven hard loss bound. Overshoot
    (excess_cum - eps_risk) and the one-window decision lag are recorded.
  * Delta_net is an OBSERVATIONAL estimate with background-trend difference
    correction (residual assumptions A1-A4); it does NOT claim causal
    isolation. Classification uses eps_prac (calibration-provisional).
    Any flag (hold anomaly / layout change / contamination / abort) or any
    missing segment rate -> UNKNOWN. No confidence-interval claims.

Units registry: spans/bcost/eps in us (int, from reduced MAX series); steps
are true-forward counts (int); rates are us/step (float); g is us/step^2.
Cross-rank aggregation is the controller's job (MAX for timings, MIN==MAX
lockstep audit for step counts) -- this module only consumes reduced values.
"""

S3_EVENTS = frozenset({
    "urgent_shrink", "confirmed_shrink", "forced_shrink_uncertain",
    "urgent_shrink_capped", "w_desync_freeze", "confirm_fail_freeze",
})

EV_GATE_GROW = "grow_gated_probe"
EV_PROBE_START = "probe_start"
EV_PROBE_START_AAA = "probe_start_aaa"
EV_PROBE_FALLBACK = "probe_fallback"
EV_PROBE_FALLBACK_AAA = "probe_fallback_aaa"
EV_PROBE_ABORT = "probe_abort_risk"
EV_COMMIT_ADMIT = "probe_commit_admit"

_SEGMENTS = ("REF", "CAND", "FB")


class ProbeAdmission(object):

    def __init__(self, mode="off", r=12, n_ref=4, n_cand=3, n_fb=4,
                 w_base=16, w_cand=32, eps_risk_us=2000000,
                 eps_prac_frac=0.01, force_abort_at=0, commit_enabled=False,
                 calib_trigger=True, dwell_grow=2, k_tb=8):
        if mode not in ("off", "gate", "aaa"):
            raise ValueError("probe mode must be off|gate|aaa, got %r" % mode)
        if mode != "off":
            if int(w_base) != 16 or int(w_cand) != 32:
                raise ValueError(
                    "v1 probe is frozen at W16->W32->W16 (W64 not authorized); "
                    "got w_base=%r w_cand=%r" % (w_base, w_cand))
            if int(r) < 1 or int(n_ref) < 1 or int(n_cand) < 1 or int(n_fb) < 1:
                raise ValueError("probe segment budgets must be >= 1")
        self.mode = mode
        self.r = int(r)
        self.n_ref = int(n_ref)
        self.n_cand = int(n_cand)
        self.n_fb = int(n_fb)
        self.w_base = int(w_base)
        self.w_cand = int(w_cand)
        self.eps_risk_us = int(eps_risk_us)
        self.eps_prac_frac = float(eps_prac_frac)
        self.force_abort_at = int(force_abort_at)
        self.commit_enabled = bool(commit_enabled)
        self.calib_trigger = bool(calib_trigger)
        self.dwell_grow = int(dwell_grow)
        self.k_tb = int(k_tb)          # diagnostic only in calibration (T-b recorded, not gating)

        # -- state (mutated ONLY in advance()) --
        self.state = "ARMED"           # ARMED|REF|CAND|FB|DONE|ABORTED
        self.license_used = False      # at-most-one probe event per run
        self.trigger_window = None
        self.pending_abort = False     # set in advance(), acted on in next propose()
        self.abort_reason = None       # "eps_risk" | "force_abort_at"
        self.excess_cum_us = 0
        self.overshoot_us = 0
        self.contaminated = False
        self.eval_window = None
        self.eval = None               # dict once finalized (DONE or ABORTED)
        self.admission_applied = False  # production deferred commit (disabled in calib)
        self.segments = {s: [] for s in _SEGMENTS}
        self.seg_flags = {s: set() for s in _SEGMENTS}
        self.seg_layout_ver = {s: None for s in _SEGMENTS}
        self.events = []               # audit trail: (window_id, kind, detail)
        self.bcost_series = []         # (window_id, bcost_us) every window -- T-b diagnostic
        self._cum_steps = 0            # cumulative reduced steps across all windows
        self._tentative = None         # propose() -> advance() handoff

    # ------------------------------------------------------------------ util
    def cand_window_target(self):
        """CAND segment length in WINDOWS: gate/aaa forward budgets must
        correspond (author point 4): gate 3x W32 = 96 steps; aaa 6x W16."""
        if self.mode == "aaa":
            return self.n_cand * self.w_cand // self.w_base
        return self.n_cand

    def _seg_for_state(self, state):
        return state if state in _SEGMENTS else None

    def _rate(self, seg):
        """Segment rate = sum(span_us) / sum(steps) over recorded windows.
        Consumes REDUCED (MAX-span, lockstep-steps) values only."""
        entries = self.segments[seg]
        if not entries:
            return None
        steps = sum(e["steps"] for e in entries)
        if steps <= 0:
            return None
        return float(sum(e["span_us"] for e in entries)) / float(steps)

    def _mid(self, seg):
        """Segment midpoint in cumulative-step space."""
        entries = self.segments[seg]
        if not entries:
            return None
        steps = sum(e["steps"] for e in entries)
        return entries[0]["cum_steps_before"] + steps / 2.0

    def flags_all(self):
        out = set()
        for s in _SEGMENTS:
            for f in self.seg_flags[s]:
                out.add("%s:%s" % (s, f))
        if self.contaminated:
            out.add("contaminated_s3")
        if self.abort_reason:
            out.add("abort:%s" % self.abort_reason)
        return sorted(out)

    # -------------------------------------------------------------- propose
    def propose(self, window_id, rec):
        """Phase-2 (pre-hash) proposal. PURE w.r.t. segment/state data.

        Returns {"action": str, "override": None|{"W_after","event","reason"},
                 "directive": None|dict}. 'directive' is applied by the
        controller to the detector ctrl ONLY on commit (phase 4):
          {"restore": True}                -> ctrl.restore(snap)
          {"set_W": int}                   -> ctrl.W = ...
          {"dwell": int, "dwell_dir": str}  -> ctrl.dwell / dwell_dir
        """
        noop = {"action": "", "override": None, "directive": None}
        if self.mode == "off":
            self._tentative = None
            return noop
        ev = rec["event"]
        w_before = int(rec["W_before"])
        w_after = int(rec["W_after"])
        s3 = ev in S3_EVENTS
        gate_needed = w_after > w_before   # whole-run growth gating (mode != off)

        def _gate_override():
            return {"W_after": w_before, "event": EV_GATE_GROW,
                    "reason": "probe_growth_gating"}

        def _stash(action, override, directive, trigger=False):
            self._tentative = {"action": action, "trigger": trigger,
                               "override": override}
            return {"action": action, "override": override,
                    "directive": directive}

        # S3 during an active segment -> detector takeover, no override.
        if s3 and self.state in _SEGMENTS:
            return _stash("takeover_s3", None, None)

        # Pending risk abort (detected at the previous CAND window close from
        # reduced data): fall back to W16 now. One-window decision lag is
        # inherent (the window during which this fires already ran at W32)
        # and is recorded as lag_window in the CAND segment.
        if self.pending_abort and self.state == "CAND":
            ov = {"W_after": self.w_base, "event": EV_PROBE_ABORT,
                  "reason": "probe_risk_abort:%s" % self.abort_reason}
            d = {"restore": True, "set_W": self.w_base,
                 "dwell": self.dwell_grow, "dwell_dir": "grow"}
            return _stash("probe_abort_risk", ov, d)

        if self.state == "ARMED":
            trig = (self.calib_trigger and not self.license_used
                    and window_id >= self.r and not s3
                    and int(rec.get("alert_k", 0)) == 0)
            if trig and gate_needed:
                return _stash("trigger_gate_grow", _gate_override(),
                              {"restore": True}, trigger=True)
            if trig:
                return _stash("trigger", None, None, trigger=True)
            if gate_needed:
                return _stash("gate_grow", _gate_override(), {"restore": True})
            return noop

        if self.state == "REF":
            if len(self.segments["REF"]) + 1 >= self.n_ref:
                # this window completes REF -> start CAND at its close
                if self.mode == "aaa":
                    ov = {"W_after": self.w_base, "event": EV_PROBE_START_AAA,
                          "reason": "probe_segment_transition"}
                    d = {"restore": True, "dwell": 0, "dwell_dir": None}
                    return _stash("probe_start_aaa", ov, d)
                ov = {"W_after": self.w_cand, "event": EV_PROBE_START,
                      "reason": "probe_segment_transition"}
                d = {"restore": True, "set_W": self.w_cand,
                     "dwell": 0, "dwell_dir": None}
                return _stash("probe_start", ov, d)
            if gate_needed:
                return _stash("gate_grow", _gate_override(), {"restore": True})
            return noop

        if self.state == "CAND":
            if len(self.segments["CAND"]) + 1 >= self.cand_window_target():
                # this window completes CAND -> fall back at its close
                if self.mode == "aaa":
                    ov = {"W_after": self.w_base, "event": EV_PROBE_FALLBACK_AAA,
                          "reason": "probe_segment_transition"}
                    d = {"restore": True, "dwell": self.dwell_grow,
                         "dwell_dir": "grow"}
                    return _stash("probe_fallback_aaa", ov, d)
                ov = {"W_after": self.w_base, "event": EV_PROBE_FALLBACK,
                      "reason": "probe_segment_transition"}
                d = {"restore": True, "set_W": self.w_base,
                     "dwell": self.dwell_grow, "dwell_dir": "grow"}
                return _stash("probe_fallback", ov, d)
            if gate_needed:
                return _stash("gate_grow", _gate_override(), {"restore": True})
            return noop

        if self.state == "FB":
            if gate_needed:
                return _stash("gate_grow", _gate_override(), {"restore": True})
            return noop

        if self.state == "DONE":
            # production deferred commit (ONE window after EVAL); disabled in
            # calibration (commit_enabled=False) -- two arms never admitted.
            if (self.commit_enabled and not self.admission_applied
                    and self.eval is not None
                    and self.eval.get("classification") == "favorable"
                    and self.eval_window is not None
                    and window_id == self.eval_window + 1):
                ov = {"W_after": self.w_cand, "event": EV_COMMIT_ADMIT,
                      "reason": "probe_deferred_commit"}
                d = {"restore": True, "set_W": self.w_cand,
                     "dwell": 0, "dwell_dir": None}
                return _stash("commit_admit", ov, d)
            if gate_needed:
                return _stash("gate_grow", _gate_override(), {"restore": True})
            return noop

        # ABORTED (terminal): whole-run growth gating still applies.
        if gate_needed:
            return _stash("gate_grow", _gate_override(), {"restore": True})
        return noop

    # -------------------------------------------------------------- advance
    def advance(self, window_id, committed, steps, span_us, bcost_us,
                true_steps_total, p2l_version):
        """Phase-4 mutator. Consumes the reduced series (span MAX, lockstep
        steps) so every rank mutates identically. Returns
        {"directive": ..., "action": ..., "seg": ...} for controller use."""
        act = self._tentative or {"action": "", "trigger": False,
                                  "override": None}
        self._tentative = None
        if self.mode == "off":
            return {"directive": None, "action": "", "seg": ""}
        self.bcost_series.append((int(window_id), int(bcost_us)))
        seg = self._seg_for_state(self.state)
        entry = None
        if seg is not None:
            # layout fingerprint/version consistency within the segment
            if self.seg_layout_ver[seg] is None:
                self.seg_layout_ver[seg] = int(p2l_version)
            elif int(p2l_version) != self.seg_layout_ver[seg]:
                self.seg_flags[seg].add("layout_change")
            entry = {
                "window_id": int(window_id),
                "steps": int(steps),
                "span_us": int(span_us),
                "bcost_us": int(bcost_us),
                "true_steps_end": int(true_steps_total),
                "cum_steps_before": int(self._cum_steps),
                "committed": bool(committed),
            }
            if act["action"] == "probe_abort_risk":
                entry["lag_window"] = True   # ran at W32 after abort detected
            self.segments[seg].append(entry)
        self._cum_steps += int(steps)

        directive = None
        if not committed:
            if seg is not None:
                self.seg_flags[seg].add("hold_anomaly")
            self.events.append((int(window_id), "hold", seg or ""))
            return {"directive": None, "action": act["action"], "seg": seg or ""}

        action = act["action"]
        directive = None

        if action == "takeover_s3" and self.state in _SEGMENTS:
            self.contaminated = True
            self.state = "ABORTED"
            self.events.append((int(window_id), "s3_contamination", seg or ""))
            self._finalize_eval(int(window_id))
            return {"directive": None, "action": action, "seg": seg or ""}

        if self.state == "ARMED":
            if act.get("trigger"):
                self.license_used = True
                self.trigger_window = int(window_id)
                self.state = "REF"
                self.events.append((int(window_id), "trigger", ""))
                directive = None  # trigger itself changes no ctrl state
                if action == "trigger_gate_grow":
                    directive = {"restore": True}
            elif action == "gate_grow":
                directive = {"restore": True}

        elif self.state == "REF":
            if action in ("probe_start", "probe_start_aaa"):
                self.state = "CAND"
                self.events.append((int(window_id), "cand_start", action))
                directive = self._directive_for(action)
            elif action == "gate_grow":
                directive = {"restore": True}

        elif self.state == "CAND":
            if action == "probe_abort_risk":
                self.state = "FB"
                self.events.append((int(window_id), "abort_fallback",
                                    self.abort_reason or ""))
                directive = self._directive_for(action)
            elif action in ("probe_fallback", "probe_fallback_aaa"):
                self.state = "FB"
                self.events.append((int(window_id), "fb_start", action))
                directive = self._directive_for(action)
                self._risk_check(int(window_id), finalize_only=True)
            else:
                if action == "gate_grow":
                    directive = {"restore": True}
                self._risk_check(int(window_id))

        elif self.state == "FB":
            if action == "gate_grow":
                directive = {"restore": True}
            if len(self.segments["FB"]) >= self.n_fb:
                self.state = "DONE"
                self._finalize_eval(int(window_id))
                self.events.append((int(window_id), "eval",
                                    self.eval["classification"] if self.eval else "?"))

        elif self.state == "DONE":
            if action == "commit_admit":
                self.admission_applied = True
                directive = self._directive_for(action)
                self.events.append((int(window_id), "commit_admit", ""))
            elif action == "gate_grow":
                directive = {"restore": True}

        elif self.state == "ABORTED":
            if action == "gate_grow":
                directive = {"restore": True}

        return {"directive": directive, "action": action, "seg": seg or ""}

    def _directive_for(self, action):
        if action in ("probe_start",):
            return {"restore": True, "set_W": self.w_cand,
                    "dwell": 0, "dwell_dir": None}
        if action == "probe_start_aaa":
            return {"restore": True, "dwell": 0, "dwell_dir": None}
        if action in ("probe_fallback", "probe_abort_risk"):
            return {"restore": True, "set_W": self.w_base,
                    "dwell": self.dwell_grow, "dwell_dir": "grow"}
        if action == "probe_fallback_aaa":
            return {"restore": True, "dwell": self.dwell_grow,
                    "dwell_dir": "grow"}
        if action == "commit_admit":
            return {"restore": True, "set_W": self.w_cand,
                    "dwell": 0, "dwell_dir": None}
        return None

    # ----------------------------------------------------------- risk/eval
    def _risk_check(self, window_id, finalize_only=False):
        """eps_risk = ABORT THRESHOLD (not a hard loss bound). excess_cum =
        total_cand_steps * max(0, rate_cand_running - rate_ref), from the
        reduced series. Overshoot = max(0, excess_cum - eps_risk) recorded;
        the one-window decision lag is inherent and logged via lag_window."""
        rate_ref = self._rate("REF")
        rate_cand = self._rate("CAND")
        if rate_ref is None or rate_cand is None:
            return
        steps = sum(e["steps"] for e in self.segments["CAND"])
        self.excess_cum_us = int(steps * max(0.0, rate_cand - rate_ref))
        self.overshoot_us = max(0, self.excess_cum_us - self.eps_risk_us)
        if finalize_only or self.mode != "gate":
            return
        n_cand = len(self.segments["CAND"])
        forced = (self.force_abort_at > 0 and n_cand >= self.force_abort_at)
        if not finalize_only and (self.excess_cum_us > self.eps_risk_us or forced):
            if not self.pending_abort:
                self.pending_abort = True
                if forced and self.excess_cum_us <= self.eps_risk_us:
                    self.abort_reason = "force_abort_at"
                else:
                    self.abort_reason = "eps_risk"
                self.events.append((int(window_id), "risk_abort_set",
                                    "%s excess=%d overshoot=%d ncand=%d" % (
                                        self.abort_reason, self.excess_cum_us,
                                        self.overshoot_us, n_cand)))

    def _finalize_eval(self, window_id):
        """Observational Delta_net with background-trend difference
        correction. NOT causal isolation: residual assumptions A1 (drift
        near-linear over the span), A2 (no regime event inside the probe),
        A3 (stream-slice heterogeneity slow), A4 (layout stable in segment
        or flagged). Any flag / abort / contamination / missing rate ->
        UNKNOWN. Descriptive point values only; coverage column stays
        BLANK (no confidence claim -- withdrawn per author point 2)."""
        self.eval_window = int(window_id)
        flags = self.flags_all()
        rate_ref = self._rate("REF")
        rate_cand = self._rate("CAND")
        rate_fb = self._rate("FB")
        mid_ref, mid_cand, mid_fb = self._mid("REF"), self._mid("CAND"), self._mid("FB")
        g = delta = eps_prac = pred = None
        if (rate_ref is not None and rate_cand is not None and rate_fb is not None
                and mid_ref is not None and mid_cand is not None and mid_fb is not None
                and (mid_fb - mid_ref) > 0):
            g = (rate_fb - rate_ref) / (mid_fb - mid_ref)
            pred = rate_ref + g * (mid_cand - mid_ref)
            delta = rate_cand - pred
            eps_prac = self.eps_prac_frac * rate_ref
        if flags or delta is None:
            cls = "UNKNOWN"
        elif delta <= -eps_prac:
            cls = "favorable"
        elif delta >= eps_prac:
            cls = "unfavorable"
        else:
            cls = "UNKNOWN"
        self.eval = {
            "rate_ref_us_per_step": rate_ref,
            "rate_cand_us_per_step": rate_cand,
            "rate_fb_us_per_step": rate_fb,
            "g_us_per_step2": g,
            "predicted_cand_us_per_step": pred,
            "delta_net_us_per_step": delta,
            "delta_net_rel": (delta / rate_ref) if (delta is not None and rate_ref) else None,
            "eps_prac_us_per_step": eps_prac,
            "classification": cls,
            "flags": flags,
            "excess_cum_us": self.excess_cum_us,
            "overshoot_us": self.overshoot_us,
            "abort_reason": self.abort_reason,
            "contaminated": self.contaminated,
            "midpoints_steps": {"ref": mid_ref, "cand": mid_cand, "fb": mid_fb},
            "n_windows": {s: len(self.segments[s]) for s in _SEGMENTS},
            "n_steps": {s: sum(e["steps"] for e in self.segments[s]) for s in _SEGMENTS},
            "commit_eligible": bool(self.commit_enabled and cls == "favorable"
                                    and not flags),
            "estimator": "observational_ABA_diff_trend_corrected",
            "causal_isolation_claimed": False,
            "coverage_guarantee": None,   # BLANK on purpose (PRE_REG E)
        }

    # -------------------------------------------------------------- summary
    def summary(self):
        return {
            "mode": self.mode,
            "params": {"r": self.r, "n_ref": self.n_ref, "n_cand": self.n_cand,
                       "n_fb": self.n_fb, "w_base": self.w_base,
                       "w_cand": self.w_cand, "eps_risk_us": self.eps_risk_us,
                       "eps_prac_frac": self.eps_prac_frac,
                       "force_abort_at": self.force_abort_at,
                       "commit_enabled": self.commit_enabled,
                       "calib_trigger": self.calib_trigger,
                       "dwell_grow": self.dwell_grow, "k_tb": self.k_tb},
            "state": self.state,
            "license_used": self.license_used,
            "trigger_window": self.trigger_window,
            "pending_abort": self.pending_abort,
            "abort_reason": self.abort_reason,
            "excess_cum_us": self.excess_cum_us,
            "overshoot_us": self.overshoot_us,
            "contaminated": self.contaminated,
            "admission_applied": self.admission_applied,
            "eval_window": self.eval_window,
            "eval": self.eval,
            "segments": {s: self.segments[s] for s in _SEGMENTS},
            "seg_flags": {s: sorted(self.seg_flags[s]) for s in _SEGMENTS},
            "seg_layout_ver": self.seg_layout_ver,
            "events": self.events,
            "bcost_series": self.bcost_series,
            "cum_steps": self._cum_steps,
        }
