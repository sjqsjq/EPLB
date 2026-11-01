SRC = "/workspace/EPLB/OEPLB/src"
def patch(path, pairs):
    p = f"{SRC}/{path}"
    s = open(p).read()
    for old, new in pairs:
        n = s.count(old)
        assert n == 1, f"{path}: anchor count {n} != 1 for: {old[:70]!r}"
        s = s.replace(old, new)
    open(p, "w").write(s)
    print(f"PATCHED {path} ({len(pairs)} edits)")

patch("sp_state.py", [
("""            _fh = int(sp_w) if int(sp_h) < 1 else int(sp_h)""",
 """            if int(sp_h) < 0:
                raise ValueError(
                    "OEPLB_SP_H must be >=0 (0/unset => H=W); got %r"
                    % (sp_h,))
            _fh = int(sp_w) if int(sp_h) == 0 else int(sp_h)"""),
])

# config.py: move the three new validations ABOVE the mode=='off' early
# return (otherwise off-mode silently skips them), and add sp_h<0 reject.
OLD_BLOCK = """        if self.pair_rule not in ('gap', 'maxdelta'):
            raise ValueError(
                "OEPLB_PAIR_RULE must be gap|maxdelta, got %r"
                % (self.pair_rule,))
        if int(self.sp_force_s) and self.sp_mode != 'adaptive':
            raise ValueError(
                "OEPLB_SP_FORCE_S=1 (A3) requires SP_MODE=adaptive; got %r"
                % (self.sp_mode,))
        if int(self.record_all_phase) and self.sp_mode == 'off':
            raise ValueError(
                "OEPLB_RECORD_ALL_PHASE=1 (A1) requires an SP mode; got off")
"""
patch("config.py", [
(OLD_BLOCK, ""),
("""        if int(getattr(self, 'sp_rev', 0)) not in (0, 1):
            raise RuntimeError(
                "OEPLB_SP_REV must be 0|1, got %r" % (self.sp_rev,))""",
 """        if int(getattr(self, 'sp_rev', 0)) not in (0, 1):
            raise RuntimeError(
                "OEPLB_SP_REV must be 0|1, got %r" % (self.sp_rev,))
        # final_plan_20261019 arm-flag validations (BEFORE the mode=='off'
        # early return so off-mode cannot silently skip them).
        if self.pair_rule not in ('gap', 'maxdelta'):
            raise ValueError(
                "OEPLB_PAIR_RULE must be gap|maxdelta, got %r"
                % (self.pair_rule,))
        if int(self.sp_h) < 0:
            raise ValueError(
                "OEPLB_SP_H must be >=0 (0/unset => H=W); got %r"
                % (self.sp_h,))
        if int(self.sp_force_s) and self.sp_mode != 'adaptive':
            raise ValueError(
                "OEPLB_SP_FORCE_S=1 (A3) requires SP_MODE=adaptive; got %r"
                % (self.sp_mode,))
        if int(self.record_all_phase) and self.sp_mode == 'off':
            raise ValueError(
                "OEPLB_RECORD_ALL_PHASE=1 (A1) requires an SP mode; got off")"""),
])
print("FIX2_OK")
