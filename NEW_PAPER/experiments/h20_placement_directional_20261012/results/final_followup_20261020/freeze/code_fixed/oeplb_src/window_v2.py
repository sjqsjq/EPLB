"""ADW v2 single-authority window controller logic (pure Python, no torch).

Design spec: NEW_PAPER/experiments/h20_placement_directional_20261012/
results/adw_v2/adw_v2_mechanism_spec.md (mechanisms M1/M2/M3).

This module receives per-window signals that the controller computes from the
all-reduced, bit-identical fresh load tensor (z: noise-normalized change
score, h_w: recoverable excess above the indivisible-expert floor in
assignment units, n_fresh: fresh recorded assignments this window). It is
deterministic pure arithmetic on those inputs, so every rank derives an
identical decision sequence; the controller additionally cross-checks a
decision hash every 16 windows and freezes W on any mismatch.

Single authority: when window_controller_version=2 this is the ONLY code
that writes the effective sync window. All legacy mutation sites (ratio-jump
shrink, converged/volatile grow, cos_sim shift/stable grow) are short-
circuited in controller.py under the same version flag; version 1 keeps the
legacy behaviour byte-identical.

All thresholds are calibration-frozen knobs. Nothing here reads domain
labels, switch times, or any evaluation-derived quantity. No claim is made
that the controller is never slower than a fixed window: detection,
confirmation and dwell all have costs, and the net effect is an empirical
question answered by the formal comparison.
"""

W_GRID = (8, 16, 32, 64)

# Event codes (logged verbatim; keep stable across versions)
EV_INIT = "init"
EV_HOLD = "hold"
EV_HOLD_ALERT = "hold_alert"
EV_URGENT_SHRINK = "urgent_shrink"
EV_URGENT_SHRINK_CAPPED = "urgent_shrink_capped"
EV_CONFIRMED_SHRINK = "confirmed_shrink"
EV_SHRINK_BLOCKED_DWELL = "shrink_blocked_dwell"
EV_SHRINK_REJECTED_LOW_HEADROOM = "shrink_rejected_low_headroom"
EV_CONFIRMED_GROW = "confirmed_grow"
EV_GROW_BLOCKED_HEADROOM = "grow_blocked_headroom"
EV_GROW_BLOCKED_DWELL = "grow_blocked_dwell"
EV_UNCERTAIN_HOLD = "uncertain_hold"
EV_UNCERTAIN_CARRY = "uncertain_carry"
EV_FORCED_SHRINK_UNCERTAIN = "forced_shrink_uncertain"
EV_MIDBAND = "midband_hold"
EV_DESYNC_FREEZE = "w_desync_freeze"

KNOB_DEFAULTS = {
    # All placeholders; calibration MUST freeze real values before any
    # formal run. version=2 startup refuses kappa<=0 (see controller).
    "kappa": 0.0,             # noise coefficient: E[1-cos] ~ kappa / n_layer_fresh
    "z_hi": 0.0,              # change-confirmation threshold (z units)
    "z_lo": 0.0,              # stationarity threshold (z units), z_lo < z_hi
    "z_urgent": 0.0,          # immediate single-step shrink threshold
    "m_shrink": 2,            # consecutive ALERT windows to confirm shrink
    "m_grow": 3,              # consecutive stationary windows to allow grow
    "n_min": 0,               # min fresh assignments per window else UNCERTAIN
    "u_max": 4,               # consecutive UNCERTAIN windows before forced shrink
    "theta_h_low": 0.0,       # grow requires h_w < theta_h_low (assignments)
    "theta_h_act": 0.0,       # shrink requires h_w >= theta_h_act (assignments)
    "dwell_shrink": 2,        # windows blocking further shrink after a shrink
    "dwell_grow": 2,          # windows blocking further grow after a grow
    "w_uncertain_target": 16, # forced-shrink target under long UNCERTAIN
    "urgent_episode_cap": 3,  # max urgent shrinks per change episode
}


def _grid_index(w):
    return W_GRID.index(w)


class WindowControllerV2:
    """Deterministic state machine: STABLE / ALERT / UNCERTAIN / DWELL.

    One call per check window via step(). Returns a record dict; the caller
    applies record['W_after'] and logs the rest. 'carry_fresh' tells the
    controller whether to skip the end-of-window history zeroing so the next
    window accumulates on top (bounded evidence accumulation under low
    samples; at most 2 windows of evidence are ever merged).
    """

    def __init__(self, w_init, knobs=None, beta=0.5):
        if w_init not in W_GRID:
            raise ValueError(
                "ADW v2: w_init=%r not on frozen grid %r (grid reuses the "
                "existing implementation and calibration material; new grid "
                "points require a new calibration)" % (w_init, W_GRID))
        self.k = dict(KNOB_DEFAULTS)
        if knobs:
            unknown = set(knobs) - set(self.k)
            if unknown:
                raise ValueError("ADW v2: unknown knobs %r" % sorted(unknown))
            self.k.update(knobs)
        if not (0.0 <= self.k["z_lo"] < self.k["z_hi"] <= self.k["z_urgent"]):
            raise ValueError("ADW v2: require 0 <= z_lo < z_hi <= z_urgent")
        if self.k["w_uncertain_target"] not in W_GRID:
            raise ValueError("ADW v2: w_uncertain_target must be on grid")
        self.W = w_init
        self.w_init = w_init
        self.beta = beta
        # state
        self.alert_k = 0          # consecutive change-confirmed windows
        self.stable_s = 0         # consecutive stationary windows
        self.dwell = 0            # windows remaining of same-direction block
        self.dwell_dir = None     # 'shrink' | 'grow' | None
        self.uncertain_run = 0    # consecutive UNCERTAIN windows
        self.carry_windows = 0    # fresh-stat windows merged into current evidence
        self.episode_shrinks = 0  # urgent+confirmed shrinks since last grow
        self.frozen = False       # set by controller on rank-desync (safe mode)
        self.window_seq = 0

    # ---- grid helpers -------------------------------------------------
    def grid_down(self, w):
        i = _grid_index(w)
        return W_GRID[max(0, i - 1)]

    def grid_up(self, w):
        i = _grid_index(w)
        return W_GRID[min(len(W_GRID) - 1, i + 1)]

    # ---- main per-window transition -----------------------------------
    def step(self, z, h_w, n_fresh):
        """z: noise-normalized change score or None (first window / no ref);
        h_w: recoverable excess (assignments, >=0); n_fresh: fresh recorded
        assignments this window (includes any carried windows)."""
        self.window_seq += 1
        k = self.k
        w_before = self.W
        reasons = []
        carry_fresh = False

        if self.frozen:
            return self._rec(w_before, EV_DESYNC_FREEZE,
                             ["frozen_after_desync"], z, h_w, n_fresh, carry_fresh)

        # -- first window: initialize reference, hold --
        if z is None:
            return self._rec(w_before, EV_INIT, ["no_reference_yet"], z, h_w,
                             n_fresh, carry_fresh)

        # -- M1: low-sample gate (explicit UNCERTAIN state) --
        if n_fresh < k["n_min"]:
            self.uncertain_run += 1
            if self.carry_windows < 1:
                # merge next window's fresh stats with this one (bounded:
                # at most 2 windows of evidence are ever merged)
                carry_fresh = True
                self.carry_windows += 1
                event = EV_UNCERTAIN_CARRY
                reasons.append("n_fresh=%d<n_min=%d; carrying fresh stats"
                               % (n_fresh, k["n_min"]))
            else:
                event = EV_UNCERTAIN_HOLD
                reasons.append("n_fresh=%d<n_min=%d; carry exhausted"
                               % (n_fresh, k["n_min"]))
            # z/h do not drive W here; the controller does NOT update the
            # detection EWMA while uncertain (prevents small-sample pollution).
            if self.uncertain_run >= k["u_max"] and self.W > k["w_uncertain_target"]:
                self.W = k["w_uncertain_target"]
                self.uncertain_run = 0
                self.carry_windows = 0
                self.dwell = k["dwell_shrink"]
                self.dwell_dir = "shrink"
                event = EV_FORCED_SHRINK_UNCERTAIN
                reasons.append("uncertain_run>=%d -> bounded response: W<=%d"
                               % (k["u_max"], k["w_uncertain_target"]))
            return self._rec(w_before, event, reasons, z, h_w, n_fresh, carry_fresh)

        # sufficient samples: reset uncertain bookkeeping
        self.uncertain_run = 0
        self.carry_windows = 0

        # -- M2+M3: change / stationary decision with headroom gating --
        event = EV_HOLD
        actionable_change = z >= k["z_hi"] and h_w >= k["theta_h_act"]
        if z >= k["z_hi"] and h_w < k["theta_h_act"]:
            # real-looking change but no priced recoverable loss: do not act.
            # "model cannot price it" is NOT "gain is zero" -> hold the
            # window (conservative), log the rejection, keep observing.
            self.alert_k = 0
            self.stable_s = 0
            event = EV_SHRINK_REJECTED_LOW_HEADROOM
            reasons.append("z=%.2f>=z_hi but h_w=%.0f<theta_h_act=%.0f"
                           % (z, h_w, k["theta_h_act"]))
        elif z >= k["z_urgent"] and actionable_change:
            # urgent path: immediate single grid step, bypasses dwell but
            # capped per episode (bounded response, no runaway shrink).
            self.alert_k = 0
            self.stable_s = 0
            if self.episode_shrinks >= k["urgent_episode_cap"]:
                event = EV_URGENT_SHRINK_CAPPED
                reasons.append("episode_shrinks=%d at cap" % self.episode_shrinks)
            elif self.W == W_GRID[0]:
                event = EV_HOLD_ALERT
                reasons.append("z=%.2f>=z_urgent but W at floor" % z)
            else:
                self.W = self.grid_down(self.W)
                self.episode_shrinks += 1
                self.dwell = k["dwell_shrink"]
                self.dwell_dir = "shrink"
                event = EV_URGENT_SHRINK
                reasons.append("z=%.2f>=z_urgent=%.2f and h_w=%.0f>=theta_h_act"
                               % (z, k["z_urgent"], h_w))
        elif actionable_change:
            self.stable_s = 0
            self.alert_k += 1
            if self.alert_k >= k["m_shrink"]:
                if self.dwell > 0 and self.dwell_dir == "shrink":
                    event = EV_SHRINK_BLOCKED_DWELL
                    reasons.append("alert_k=%d confirmed but dwell=%d(%s)"
                                   % (self.alert_k, self.dwell, self.dwell_dir))
                elif self.W == W_GRID[0]:
                    self.alert_k = 0
                    event = EV_HOLD_ALERT
                    reasons.append("alert confirmed but W at floor")
                else:
                    self.W = self.grid_down(self.W)
                    self.episode_shrinks += 1
                    self.alert_k = 0
                    self.dwell = k["dwell_shrink"]
                    self.dwell_dir = "shrink"
                    event = EV_CONFIRMED_SHRINK
                    reasons.append("alert_k>=%d and h_w=%.0f>=theta_h_act"
                                   % (k["m_shrink"], h_w))
            else:
                event = EV_HOLD_ALERT
                reasons.append("alert_k=%d/%d z=%.2f" % (self.alert_k, k["m_shrink"], z))
        elif z < k["z_lo"]:
            # stationary band
            self.alert_k = 0
            self.stable_s += 1
            if self.stable_s >= k["m_grow"]:
                if h_w >= k["theta_h_low"]:
                    # M2: distribution stable BUT recoverable headroom remains
                    # -> keep checking frequency (do not lengthen).
                    event = EV_GROW_BLOCKED_HEADROOM
                    reasons.append("stable_s=%d but h_w=%.0f>=theta_h_low=%.0f"
                                   % (self.stable_s, h_w, k["theta_h_low"]))
                elif self.dwell > 0 and self.dwell_dir == "grow":
                    event = EV_GROW_BLOCKED_DWELL
                    reasons.append("stable_s=%d but dwell=%d(grow)"
                                   % (self.stable_s, self.dwell))
                elif self.W == W_GRID[-1]:
                    event = EV_HOLD
                    reasons.append("stable and W at ceiling %d (bounded "
                                   "response: worst-case detection delay "
                                   "<= W_max steps)" % W_GRID[-1])
                else:
                    self.W = self.grid_up(self.W)
                    self.stable_s = 0
                    self.episode_shrinks = 0
                    self.dwell = k["dwell_grow"]
                    self.dwell_dir = "grow"
                    event = EV_CONFIRMED_GROW
                    reasons.append("stable_s>=%d and h_w=%.0f<theta_h_low"
                                   % (k["m_grow"], h_w))
            else:
                event = EV_HOLD
                reasons.append("stable_s=%d/%d" % (self.stable_s, k["m_grow"]))
        else:
            # hysteresis mid-band: z in [z_lo, z_hi) - clear alert streak,
            # keep stationary streak (neither direction is confirmed).
            self.alert_k = 0
            event = EV_MIDBAND
            reasons.append("z=%.2f in [z_lo,z_hi)" % z)

        if self.dwell > 0:
            self.dwell -= 1
            if self.dwell == 0:
                self.dwell_dir = None
        return self._rec(w_before, event, reasons, z, h_w, n_fresh, carry_fresh)

    def on_desync(self):
        """Safe mode on cross-rank decision-hash mismatch: freeze W forever at
        the DETERMINISTIC w_init (not "current W" - after a desync the current
        W may itself differ across ranks, which would break the lockstep check
        schedule and risk collective divergence; w_init is identical everywhere
        by construction)."""
        self.frozen = True
        self.W = self.w_init

    def _rec(self, w_before, event, reasons, z, h_w, n_fresh, carry_fresh):
        return {
            "window_seq": self.window_seq,
            "W_before": w_before,
            "W_after": self.W,
            "event": event,
            "reasons": reasons,
            "z": z,
            "h_w": h_w,
            "n_fresh": n_fresh,
            "alert_k": self.alert_k,
            "stable_s": self.stable_s,
            "dwell": self.dwell,
            "dwell_dir": self.dwell_dir,
            "uncertain_run": self.uncertain_run,
            "episode_shrinks": self.episode_shrinks,
            "carry_fresh": carry_fresh,
        }
