"""sp_coupling_v1 -- sampling/planning window coupling state machine.

Plan: NEW_PAPER/experiments/h20_placement_directional_20261012/results/
adw_v2p1/next_version/PLAN_sampling_planning_coupling_20261018.md
Pre-data design decisions PD-1..PD-15 (sealed BEFORE implementation):
NEW_PAPER/.../results/sp_coupling_v1/MAPPING_sp_coupling.md.

Pure Python, no torch, no collectives. The machine is a deterministic
function of its committed state plus the all-rank-reduced signals consumed
in phase 2 of the v2.1 joint-commit protocol; its proposal is folded into
the EXISTING single decision hash (PD-6), and state transitions are applied
ONLY on joint commit (phase 4). On HOLD the proposal is simply discarded:
machine state advances only on committed cycles, so held boundaries neither
advance nor reset streaks.

Modes (PD-11):
  off      -- inert; the controller keeps the legacy path bit-identical.
  fixed    -- constant (S,H,W) = (SP_S, SP_W, SP_W); no transitions; shares
              the SAME pre-forward sampling-gate code path as adaptive
              (plan Sec.6 fairness: F_sparse gets identical sparse-record
              capability, F_dense identical dense capability).
  adaptive -- TRACK/QUIET/RECOVER machine on the frozen table below.

State table (frozen pre-data, plan Sec.3.1-3.3; per governance these values
may not change after the timed campaign starts):
  TRACK   (S=1, H=16, W=16)  dense sampling, detection+planning each 16
  QUIET   (S=4, H=16, W=64)  sparse sampling, planning every 4th H cycle
  RECOVER (S=1, H=8,  W=8)   dense everything, fastest recovery schedule
Invariant enforced at construction: W % H == 0 in every state, so W
boundaries are a strict subset of H boundaries and the per-rank collective
schedule is a fixed function of the boundary type alone:
  H-only : [detSUM, confirmMIN, confirmMAX]
  H+W    : [plannerSUM, detSUM, confirmMIN, confirmMAX]

All thresholds consumed here (z_lo, z_hi, theta_h_low, n_min) are the
calibration-FROZEN v2.1 values passed in by the controller; this module
introduces NO new numeric thresholds. The only derived quantity is
n_min_eff = n_min // S (PD-1, pre-declared; subject to the Sec.4.3
dev-data noise check, never tuned on evaluation data).
"""

import hashlib
import json
import os

SP_STATES = ("TRACK", "QUIET", "RECOVER")

# (S, H, W) per state -- frozen table (plan Sec.3, PD-11).
SP_TABLE = {
    "TRACK":   (1, 16, 16),
    "QUIET":   (4, 16, 64),
    "RECOVER": (1, 8, 8),
}

# Guardrail events (MAPPING C-table last row, PD-3): consistency/state
# guards that force QUIET exit regardless of z/h_rate. The uncertain family
# means coverage failed under the current sampling rate; confirm/desync
# events mean the joint protocol itself degraded. "hold_alert"/shrink
# events are NOT guardrails: they are ordinary change evidence and QUIET
# exit already fires on the first z>=z_hi alert.
SP_GUARDRAIL_EVENTS = frozenset({
    "uncertain_carry", "uncertain_hold", "forced_shrink_uncertain",
    "w_desync_freeze", "confirm_hold", "confirm_fail_freeze",
})

# rev1 (2026-10-18, author R2): confirmed-change / pre-existing-urgent event
# family, taken VERBATIM from the frozen window_v2.py vocabulary -- no new
# detector. hold_alert/shrink-family events are alert-band (z>=z_hi at a W
# boundary inside the cycle) or urgent (z>=z_urgent); confirmed_grow/shrink
# are v21's own two-window confirmed drift detectors. grow_blocked_* are
# opportunistic (not confirmed change) and stay OUT; uncertain_*/freeze are
# already guardrails (P1b).
SP_CHANGE_EVENTS = frozenset({
    "hold_alert", "urgent_shrink", "urgent_shrink_capped",
    "confirmed_shrink", "shrink_blocked_dwell",
    "shrink_rejected_low_headroom", "confirmed_grow",
})

# rev1 (author R3): sustained exit confirmation count. Matches the frozen
# confirmation conventions already deployed (window_shift_confirm_windows=2,
# v21_dwell_shrink/grow=2/2). Module constant, NOT an online knob.
SP_QUIET_EXIT_CONFIRM = 2

_HASH_MOD = 1 << 58   # same namespace bound as window_v2p1.decision_hash


def sp_record_gate(seed, sp_fwd, s):
    """PD-9: deterministic S-gate on the all-rank-consistent global forward
    number. block = sp_fwd // s; pos = blake2b(seed || block) % s; the
    forward is a record opportunity iff sp_fwd % s == pos. The sampled
    position inside each block comes from the frozen seed and the block
    number, so the sampled slot cannot phase-lock with any mod-s periodic
    backend pattern. s <= 1 -> every forward is an opportunity. Every rank
    computes the identical answer without communication (sp_fwd advances in
    lockstep, idle ranks included)."""
    if s <= 1:
        return True
    block = sp_fwd // s
    digest = hashlib.blake2b(
        ("%d:%d" % (seed, block)).encode("utf-8"), digest_size=8).digest()
    pos = int.from_bytes(digest, "big") % s
    return (sp_fwd % s) == pos


class SpController:
    """TRACK/QUIET/RECOVER coupling machine + fixed-arm constant schedule.

    Lifecycle per H boundary (controller side):
      phase 2: prop = propose(sig)          -- pure, no mutation
               h ^= proposal_hash(prop)     -- folded into decision hash
      phase 3: confirm MIN/MAX (existing collectives, unchanged)
      phase 4: commit -> apply(prop)        -- ONLY mutation point
               hold   -> nothing (prop discarded)
    """

    def __init__(self, mode="off", sp_s=1, sp_w=16, seed=20261018,
                 quiet_streak=3, recover_min_cycles=2,
                 z_lo=0.0, z_hi=0.0, theta_h_low=0.0, n_min=0,
                 theta_h_act=0.0, rev=0, sp_h=0, force_s=0):
        if mode not in ("off", "fixed", "adaptive"):
            raise ValueError(
                "sp_mode must be off|fixed|adaptive, got %r" % (mode,))
        self.mode = mode
        self.seed = int(seed)
        self.quiet_streak = int(quiet_streak)
        self.recover_min_cycles = int(recover_min_cycles)
        self.z_lo = float(z_lo)
        self.z_hi = float(z_hi)
        self.theta_h_low = float(theta_h_low)
        self.n_min_full = int(n_min)
        self.theta_h_act = float(theta_h_act)
        self.rev = int(rev)
        if self.rev not in (0, 1):
            raise ValueError("sp rev must be 0|1, got %r" % (rev,))
        # AERO-RG recovery-admission policy (2026-10-27, author-ordered
        # mechanism change; recovery_admission_e2e_20261027 PROTOCOL.md):
        #   OEPLB_SP_RG unset/0 -> AERO-base: decisions AND proposal hashes
        #     are byte-identical to the pre-patch frozen version (ba572f93).
        #   OEPLB_SP_RG=1 -> AERO-RG: in QUIET (rev1 only) a LONE
        #     confirmed_grow no longer counts as change evidence (it means
        #     stable-window growth, not a load change). Every other exit
        #     condition, priority and guardrail is untouched; the elif
        #     chain is NOT short-circuited (no early return).
        self.rg = (os.environ.get("OEPLB_SP_RG", "0") == "1")
        if self.rg and not (self.rev == 1 and mode == "adaptive"):
            raise ValueError(
                "OEPLB_SP_RG=1 (AERO-RG) requires rev=1 and mode=adaptive "
                "(got rev=%r mode=%r)" % (self.rev, mode))
        if self.rev == 1 and mode == "adaptive":
            # rev1 hysteresis band: both edges are the frozen calibration
            # pair (theta_r_low, theta_r_act); ordering is validated, the
            # values are never re-derived here.
            if not (self.theta_h_act > self.theta_h_low > 0.0):
                raise ValueError(
                    "rev1 requires frozen theta_h_low < theta_h_act "
                    "(calibration pair); got %r / %r"
                    % (self.theta_h_low, self.theta_h_act))
        if mode == "adaptive":
            if not (0.0 < self.z_lo < self.z_hi):
                raise ValueError(
                    "sp adaptive requires frozen 0 < z_lo < z_hi (got "
                    "z_lo=%r z_hi=%r)" % (self.z_lo, self.z_hi))
            if self.theta_h_low <= 0.0:
                raise ValueError(
                    "sp adaptive requires frozen theta_h_low > 0 (rate units)")
            if self.n_min_full <= 0:
                raise ValueError(
                    "sp adaptive requires frozen n_min > 0 (PD-1 derives "
                    "n_min_eff = n_min // S from it)")
            if self.quiet_streak < 1 or self.recover_min_cycles < 1:
                raise ValueError(
                    "quiet_streak/recover_min_cycles must be >= 1")
        for _st, (_s, _h, _w) in SP_TABLE.items():
            if _s < 1 or _h < 1 or _w < _h or _w % _h != 0:
                raise ValueError(
                    "SP_TABLE state %r=(%d,%d,%d) violates S>=1, H>=1, "
                    "W>=H, W%%H==0" % (_st, _s, _h, _w))
        self._fixed = None
        self._force_s = int(force_s)
        if mode == "fixed":
            if int(sp_s) < 1 or int(sp_w) < 1:
                raise ValueError(
                    "fixed mode requires SP_S>=1 and SP_W>=1 (got S=%r W=%r)"
                    % (sp_s, sp_w))
            # final_plan_20261019 (F arm): optional SP_H decouples H from W
            # for fixed schedules. sp_h=0/unset keeps the sealed H=W behavior
            # bit-identically (F_dense compatibility).
            if int(sp_h) < 0:
                raise ValueError(
                    "OEPLB_SP_H must be >=0 (0/unset => H=W); got %r"
                    % (sp_h,))
            _fh = int(sp_w) if int(sp_h) == 0 else int(sp_h)
            if _fh < 1 or int(sp_w) < _fh or int(sp_w) % _fh != 0:
                raise ValueError(
                    "fixed mode requires H>=1, W>=H, W%%H==0 "
                    "(got H=%r W=%r)" % (_fh, sp_w))
            self._fixed = (int(sp_s), _fh, int(sp_w))  # (S,H,W)
        self.state = "FIXED" if mode == "fixed" else "TRACK"
        self.streak = 0            # consecutive quiet valid H cycles (TRACK)
        self.recover_cycles = 0    # consecutive clean H8 cycles (RECOVER)
        self.exit_streak = 0       # rev1: consecutive QUIET H cycles with
                                   # h_rate >= theta_out (0 under rev0)
        self.transitions = []      # committed transitions (run timeline)
        self.residency = {}        # committed H cycles per state

    # ---- schedule -----------------------------------------------------
    @property
    def S(self):
        # final_plan_20261019 A3 (P_denseSample): S forced to 1 in adaptive
        # mode; H/W state rules untouched. Default force_s=0 -> sealed path.
        if self._force_s and self.mode == "adaptive":
            return 1
        if self._fixed is not None:
            return self._fixed[0]
        return SP_TABLE[self.state][0] if self.mode == "adaptive" else 1

    @property
    def H(self):
        if self._fixed is not None:
            return self._fixed[1]
        return SP_TABLE[self.state][1] if self.mode == "adaptive" else 1

    @property
    def W(self):
        if self._fixed is not None:
            return self._fixed[2]
        return SP_TABLE[self.state][2] if self.mode == "adaptive" else 1

    def n_min_eff(self):
        """PD-1: coverage gate scaled by the sampling rate, pre-declared as
        n_min // S. NEVER inflated by samples x S; whether this scale
        survives S4 background variability is the Sec.4.3 dev-data check
        (one-time freeze from dev materials only), not an online knob. If
        the check shows QUIET exiting on coverage often, that is reported
        honestly -- the gate is not lowered to force residency."""
        return max(1, self.n_min_full // max(1, self.S))

    def gate(self, sp_fwd):
        return sp_record_gate(self.seed, sp_fwd, self.S)

    # ---- proposal (phase 2, pure) -------------------------------------
    def propose(self, sig):
        """sig keys (all all-rank-reduced or committed-state quantities):
          z          float|None  noise-normalized change score (None=init)
          h_rate     float       headroom rate h_w/n_fresh (opportunity proxy)
          n_fresh    int         fresh recorded assignments this H cycle
          event      str         window-ctrl event this cycle (guardrails)
          swap_pending  bool     executor holds an unfinished plan
          swaps_started int      plan starts since the PREVIOUS H boundary
          plan_opp   bool        a W boundary fired within this H cycle
        Returns a prop dict. NO mutation (PD-6: state advances only via
        apply() on joint commit)."""
        if self.mode != "adaptive":
            return {"transition": False, "state_after": self.state,
                    "streak_after": self.streak,
                    "recover_cycles_after": self.recover_cycles,
                    "reasons": [], "valid_cycle": False,
                    "quiet_cond": False, "clean_cycle": False,
                    "exit_streak_after": self.exit_streak,
                    "n_min_eff": self.n_min_eff()}
        z = sig.get("z")
        h_rate = float(sig.get("h_rate") or 0.0)
        n_fresh = int(sig.get("n_fresh") or 0)
        ev = sig.get("event") or ""
        nme = self.n_min_eff()
        coverage = n_fresh >= nme
        guardrail = ev in SP_GUARDRAIL_EVENTS
        # PD-2: a VALID H cycle = coverage passed, z defined, no guardrail.
        valid = (z is not None) and coverage and not guardrail
        # QUIET-entry condition (PD-2, MAPPING C rows 1-4): stable band +
        # low opportunity margin + coverage + no pending migration + no
        # accepted swap since the previous H boundary. plan_opp=False is
        # logged separately and never COUNTED as "no action" evidence; in
        # TRACK H==W so every entry-relevant cycle has a planning
        # opportunity, and in QUIET/RECOVER the entry streak does not run.
        quiet_cond = (valid
                      and z < self.z_lo
                      and h_rate < self.theta_h_low
                      and not sig.get("swap_pending")
                      and int(sig.get("swaps_started") or 0) == 0)
        clean = (valid and z < self.z_lo and h_rate < self.theta_h_low)
        reasons = []
        transition = False
        state_after = self.state
        streak_after = self.streak
        rec_after = self.recover_cycles
        exit_after = self.exit_streak
        if self.state == "TRACK":
            streak_after = self.streak + 1 if quiet_cond else 0
            if streak_after >= self.quiet_streak:
                transition = True
                state_after = "QUIET"
                streak_after = 0     # PD-5
                reasons.append(
                    "quiet_streak>=%d: z<z_lo & h_rate<theta_h_low & "
                    "coverage & no pending/accepted swaps for %d "
                    "consecutive valid H cycles"
                    % (self.quiet_streak, self.quiet_streak))
        elif self.state == "QUIET":
            if self.rev:
                # rev1 (2026-10-18, author R1/R2/R3): same-window priority
                # P1 RECOVER (confirmed change evidence / pre-existing
                # urgent conditions) > P2 TRACK (insufficient evidence) >
                # P3 TRACK (sustained low-margin failure through the
                # hysteresis band [theta_h_low, theta_h_act) with
                # SP_QUIET_EXIT_CONFIRM consecutive cycles). Full table and
                # parameter provenance: REV1_quiet_exit_revision_20261018.md.
                # AERO-RG: a LONE confirmed_grow in QUIET = stable-window
                # growth, NOT a load change -> excluded from change_ev.
                # P1a (z alert), P1b (guardrail), P2 (coverage) and P3
                # (hysteresis) are still evaluated with original priority;
                # alerts/shrink/migration evidence keeps original behavior.
                _rg_lone_grow = (self.rg
                                 and ev == "confirmed_grow"
                                 and not bool(sig.get("swap_pending"))
                                 and int(sig.get("swaps_started") or 0) <= 0)
                if _rg_lone_grow:
                    reasons.append(
                        "[AERO-RG] lone confirmed_grow ignored as change "
                        "evidence (stable-window growth, not load change); "
                        "all other QUIET exit conditions still evaluated")
                _ev_eff = "" if _rg_lone_grow else ev
                change_ev = (_ev_eff in SP_CHANGE_EVENTS
                             or bool(sig.get("swap_pending"))
                             or int(sig.get("swaps_started") or 0) > 0)
                if z is not None and z >= self.z_hi:
                    transition = True
                    state_after = "RECOVER"
                    reasons.append(
                        "P1a z=%.3f>=z_hi=%.3f first alert (pre-existing "
                        "urgent condition, unchanged) -- sampling recovers "
                        "immediately; planning action still requires the "
                        "existing alert_k confirmation" % (z, self.z_hi))
                elif guardrail:
                    transition = True
                    state_after = "RECOVER"
                    reasons.append("P1b guardrail event %r" % ev)
                elif change_ev:
                    transition = True
                    state_after = "RECOVER"
                    reasons.append(
                        "P1c/P1d confirmed change evidence: event=%r "
                        "swap_pending=%r swaps_started=%d"
                        % (ev, bool(sig.get("swap_pending")),
                           int(sig.get("swaps_started") or 0)))
                elif not coverage:
                    transition = True
                    state_after = "TRACK"
                    reasons.append(
                        "P2 n_fresh=%d<n_min_eff=%d coverage failure under "
                        "S=%d = insufficient evidence -> TRACK (rev1 R1; "
                        "NOT RECOVER)" % (n_fresh, nme, self.S))
                    exit_after = 0
                else:
                    exit_after = (self.exit_streak + 1
                                  if h_rate >= self.theta_h_act else 0)
                    if exit_after >= SP_QUIET_EXIT_CONFIRM:
                        transition = True
                        state_after = "TRACK"
                        reasons.append(
                            "P3 h_rate=%.6f>=theta_out=%.6f for %d "
                            "consecutive QUIET cycles (sustained low-margin "
                            "failure through the hysteresis band) -> TRACK "
                            "(rev1 R1/R3)"
                            % (h_rate, self.theta_h_act, exit_after))
                        exit_after = 0
            else:
                # PD-3 (v1, frozen): exit on ANY of these; low h_rate NEVER
                # blocks recovery.
                if z is not None and z >= self.z_hi:
                    transition = True
                    reasons.append(
                        "z=%.3f>=z_hi=%.3f first alert -- sampling recovers "
                        "immediately; planning action still requires the "
                        "existing alert_k confirmation" % (z, self.z_hi))
                elif h_rate >= self.theta_h_low:
                    transition = True
                    reasons.append(
                        "h_rate=%.6f>=theta_h_low=%.6f opportunity proxy rose"
                        % (h_rate, self.theta_h_low))
                elif not coverage:
                    transition = True
                    reasons.append(
                        "n_fresh=%d<n_min_eff=%d coverage failure under S=%d"
                        % (n_fresh, nme, self.S))
                elif guardrail:
                    transition = True
                    reasons.append("guardrail event %r" % ev)
                if transition:
                    state_after = "RECOVER"
        elif self.state == "RECOVER":
            # PD-4: >= recover_min_cycles COMPLETE clean H8 cycles (stable
            # band + low margin + coverage). High headroom keeps RECOVER;
            # there is NO timed forced convergence.
            rec_after = self.recover_cycles + 1 if clean else 0
            if rec_after >= self.recover_min_cycles:
                transition = True
                state_after = "TRACK"
                rec_after = 0        # PD-5
                reasons.append(
                    "recover_min_cycles>=%d complete clean H%d cycles "
                    "(stable+low-margin+coverage); no timeout convergence"
                    % (self.recover_min_cycles, SP_TABLE["RECOVER"][1]))
        return {"transition": transition, "state_after": state_after,
                "streak_after": streak_after,
                "recover_cycles_after": rec_after,
                "reasons": reasons, "valid_cycle": valid,
                "quiet_cond": quiet_cond,
                "clean_cycle": (self.state == "RECOVER" and clean),
                "exit_streak_after": 0 if transition else exit_after,
                "n_min_eff": nme}

    def proposal_hash(self, prop):
        """PD-6 hash contribution. Includes the machine counters: they are
        deterministic functions of the committed history, so identical
        ranks always agree, and any divergence (however benign-looking)
        demotes the boundary to a joint HOLD instead of drifting."""
        if self.rev:
            payload = ["sp1r1rg" if self.rg else "sp1r1",
                       self.mode, self.state, self.streak,
                       self.recover_cycles, self.exit_streak,
                       bool(prop.get("transition")),
                       prop.get("state_after", self.state),
                       int(prop.get("streak_after", 0)),
                       int(prop.get("recover_cycles_after", 0)),
                       int(prop.get("exit_streak_after", 0))]
        else:
            payload = ["sp1", self.mode, self.state, self.streak,
                       self.recover_cycles, bool(prop.get("transition")),
                       prop.get("state_after", self.state),
                       int(prop.get("streak_after", 0)),
                       int(prop.get("recover_cycles_after", 0))]
        digest = hashlib.sha256(
            json.dumps(payload).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % _HASH_MOD

    # ---- commit (phase 4, ONLY mutator) --------------------------------
    def apply(self, prop, h_id=None):
        self._bump_residency()   # counts the state the cycle was served in
        s_before = self.S
        transition = bool(prop.get("transition")) and self.mode == "adaptive"
        if transition:
            old = self.state
            self.state = prop["state_after"]
            self.streak = 0            # PD-5: any transition resets streaks
            self.recover_cycles = 0
            self.exit_streak = 0       # rev1: PD-5 extended
            _rsn = list(prop.get("reasons", []))
            if self.rg:
                _rsn = ["[AERO-RG] policy=rg rev=1"] + _rsn
            self.transitions.append({
                "h_id": h_id, "from": old, "to": self.state,
                "reasons": _rsn,
                "S": self.S, "H": self.H, "W": self.W})
        else:
            self.streak = int(prop.get("streak_after", self.streak))
            self.recover_cycles = int(
                prop.get("recover_cycles_after", self.recover_cycles))
            self.exit_streak = int(
                prop.get("exit_streak_after", self.exit_streak))
        return {"transition": transition,
                "S_changed": (self.S != s_before),
                "state": self.state, "S": self.S, "H": self.H, "W": self.W}

    def _bump_residency(self):
        key = self.state
        self.residency[key] = self.residency.get(key, 0) + 1

    # ---- snapshot / restore (test + safety symmetry) -------------------
    def snapshot(self):
        return {"state": self.state, "streak": self.streak,
                "recover_cycles": self.recover_cycles,
                "exit_streak": self.exit_streak,
                "n_transitions": len(self.transitions),
                "residency": dict(self.residency)}

    def restore(self, snap):
        self.state = snap["state"]
        self.streak = snap["streak"]
        self.recover_cycles = snap["recover_cycles"]
        self.exit_streak = snap["exit_streak"]
        del self.transitions[snap["n_transitions"]:]
        self.residency = dict(snap["residency"])
