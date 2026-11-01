import sys
SRC = "/workspace/EPLB/OEPLB/src"

def patch(path, pairs):
    p = f"{SRC}/{path}"
    s = open(p).read()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{path}: anchor count {n} != 1 for: {old[:80]!r}"
        s = s.replace(old, new)
    open(p, "w").write(s)
    print(f"PATCHED {path} ({len(pairs)} edits)")

# ---------------- sp_state.py ----------------
patch("sp_state.py", [
("theta_h_act=0.0, rev=0):",
 "theta_h_act=0.0, rev=0, sp_h=0, force_s=0):"),
("""        self._fixed = None
        if mode == "fixed":
            if int(sp_s) < 1 or int(sp_w) < 1:
                raise ValueError(
                    "fixed mode requires SP_S>=1 and SP_W>=1 (got S=%r W=%r)"
                    % (sp_s, sp_w))
            self._fixed = (int(sp_s), int(sp_w), int(sp_w))  # (S,H,W), H=W""",
 """        self._fixed = None
        self._force_s = int(force_s)
        if mode == "fixed":
            if int(sp_s) < 1 or int(sp_w) < 1:
                raise ValueError(
                    "fixed mode requires SP_S>=1 and SP_W>=1 (got S=%r W=%r)"
                    % (sp_s, sp_w))
            # final_plan_20261019 (F arm): optional SP_H decouples H from W
            # for fixed schedules. sp_h=0/unset keeps the sealed H=W behavior
            # bit-identically (F_dense compatibility).
            _fh = int(sp_w) if int(sp_h) < 1 else int(sp_h)
            if _fh < 1 or int(sp_w) < _fh or int(sp_w) % _fh != 0:
                raise ValueError(
                    "fixed mode requires H>=1, W>=H, W%%H==0 "
                    "(got H=%r W=%r)" % (_fh, sp_w))
            self._fixed = (int(sp_s), _fh, int(sp_w))  # (S,H,W)"""),
("""    @property
    def S(self):
        if self._fixed is not None:
            return self._fixed[0]
        return SP_TABLE[self.state][0] if self.mode == "adaptive" else 1""",
 """    @property
    def S(self):
        # final_plan_20261019 A3 (P_denseSample): S forced to 1 in adaptive
        # mode; H/W state rules untouched. Default force_s=0 -> sealed path.
        if self._force_s and self.mode == "adaptive":
            return 1
        if self._fixed is not None:
            return self._fixed[0]
        return SP_TABLE[self.state][0] if self.mode == "adaptive" else 1"""),
])

# ---------------- config.py ----------------
patch("config.py", [
("    sp_rev: int = 0              # 0 = v1 frozen; 1 = rev1 QUIET-exit",
 """    sp_rev: int = 0              # 0 = v1 frozen; 1 = rev1 QUIET-exit
    sp_h: int = 0                # final_plan F arm: fixed-mode H (0 => H=W)
    sp_force_s: int = 0          # final_plan A3: force S=1 (adaptive only)
    record_all_phase: int = 0    # final_plan A1: record decode forwards too
    no_commit: int = 0           # final_plan A4: build plans, never commit
    pair_rule: str = 'gap'       # final_plan A2: 'gap' (sealed) | 'maxdelta'"""),
("            sp_rev=int(os.environ.get('OEPLB_SP_REV', '0')),",
 """            sp_rev=int(os.environ.get('OEPLB_SP_REV', '0')),
            sp_h=int(os.environ.get('OEPLB_SP_H', '0')),
            sp_force_s=int(os.environ.get('OEPLB_SP_FORCE_S', '0')),
            record_all_phase=int(os.environ.get('OEPLB_RECORD_ALL_PHASE', '0')),
            no_commit=int(os.environ.get('OEPLB_NO_COMMIT', '0')),
            pair_rule=os.environ.get('OEPLB_PAIR_RULE', 'gap'),"""),
("""                "adaptive QUIET-exit routing only; got %r)" % (self.sp_mode,))""",
 """                "adaptive QUIET-exit routing only; got %r)" % (self.sp_mode,))
        if self.pair_rule not in ('gap', 'maxdelta'):
            raise ValueError(
                "OEPLB_PAIR_RULE must be gap|maxdelta, got %r"
                % (self.pair_rule,))
        if int(self.sp_force_s) and self.sp_mode != 'adaptive':
            raise ValueError(
                "OEPLB_SP_FORCE_S=1 (A3) requires SP_MODE=adaptive; got %r"
                % (self.sp_mode,))
        if int(self.record_all_phase) and self.sp_mode == 'off':
            raise ValueError(
                "OEPLB_RECORD_ALL_PHASE=1 (A1) requires an SP mode; got off")"""),
])

# ---------------- rebalancer.py ----------------
patch("rebalancer.py", [
("import torch\nimport logging",
 "import torch\nimport logging\nimport os"),
("""                if hot_cands[0][0] if isinstance(hot_cands[0], tuple) else lc[hot_cands[0]] <= gap:""",
 """                # final_plan_20261019 A2 (P_maxdelta): OEPLB_PAIR_RULE=maxdelta
                # takes the max-transfer pair unconditionally (same legal
                # candidate sets/acceptance/budgets; only the pair rule
                # changes). Default 'gap' => sealed fast/gap two-branch.
                if os.environ.get('OEPLB_PAIR_RULE', 'gap') == 'maxdelta' or (hot_cands[0][0] if isinstance(hot_cands[0], tuple) else lc[hot_cands[0]] <= gap):"""),
])

# ---------------- controller.py ----------------
patch("controller.py", [
('            rev=int(getattr(cfg, "sp_rev", 0)))',
 '''            rev=int(getattr(cfg, "sp_rev", 0)),
            sp_h=int(getattr(cfg, "sp_h", 0)),
            force_s=int(getattr(cfg, "sp_force_s", 0)))'''),
("""        self._sample_interval = cfg.sample_interval
        self._should_record_this_batch = False""",
 """        self._sample_interval = cfg.sample_interval
        self._should_record_this_batch = False
        # final_plan_20261019 arm flags (all default OFF => sealed rev1
        # behavior bit-identical; see results/final_plan_20261019/C_configs)
        self._record_all_phase = bool(int(getattr(cfg, "record_all_phase", 0)))
        self._no_commit = bool(int(getattr(cfg, "no_commit", 0)))
        self._nocommit_planned_ops = 0"""),
("""        self._should_record_this_batch = bool(
            gate and (not is_idle) and is_prefill)""",
 """        self._should_record_this_batch = bool(
            gate and (not is_idle)
            and (is_prefill or self._record_all_phase))"""),
("""            self._plateau_count = 0
            self.async_executor.begin(plan)
            if self._sp_active:
                self._sp_swap_started_h += 1
                self._sp_swaps_total += 1""",
 """            self._plateau_count = 0
            if self._no_commit:
                # final_plan_20261019 A4 (P_noCommit): identical record/decide/
                # planbuild path; ONLY the physical commit is suppressed. The
                # executor is never engaged, so try_finish() has no pending
                # plan and no mapping/p2l commit occurs. swaps_started stays 0
                # (physical truth); planned ops are tallied separately.
                self._nocommit_planned_ops += len(plan)
                logger.info(
                    "[PB-OEPLB-NOCOMMIT] window#%d: plan built, %d op(s) "
                    "suppressed (observe-only arm)",
                    self.window_count, len(plan))
            else:
                self.async_executor.begin(plan)
                if self._sp_active:
                    self._sp_swap_started_h += 1
                    self._sp_swaps_total += 1"""),
('                "swaps_started_total": self._sp_swaps_total,',
 '''                "swaps_started_total": self._sp_swaps_total,
                "nocommit_planned_ops_total": self._nocommit_planned_ops,'''),
])
print("ALL_PATCHES_OK")
