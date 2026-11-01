# UNRECOVERED: EFFECT_REPORT_old_sglang.md

Status: NOT RECOVERED by this sandbox (2026-10-25).

Why: this artifact was created AFTER the last git commit of the dsv4 tree
(HEAD 5df703d6a951b8a42fe86d2dd99dc03faf76d399, 2026-10-22T18:05:34Z, branch
sp_coupling_dev) and was never committed. It existed only on disk (deleted
2026-10-25 by author order) and inside the client-host session transcript:
  /home/admin/.claude/projects/-home-admin--aone-cloud-cli-runner-cwd-98046363a249-workspace/17bd8aca-3c5e-4017-ab8d-079c16661419.jsonl
That transcript path is on the client host; every tool in this recovery
session (mcp__local__* and all subagents) reaches only the remote GPU
container, so the transcript could not be parsed and its sha256 could not be
computed from here.

Expected content per the tasking brief (UNVERIFIED — quoted from the task
prompt, not from any recovered source; do not cite as evidence without
transcript verification):
  old-tree sglang 0.5.6.post2 clone wall-ratio results:
    e2 4.3 -> 4.1 tps ; e3 4.4 -> 4.1 tps ; e4 229.6 -> 203.5 tps
    wall-ratio deltas: e2 -6.0% , e3 -6.7% , e4 -11.4%

To recover: on a session with native shell access to the client host, run a
stream-parsing extraction over the JSONL above for keys 'EFFECT_REPORT_old'
and the numbers 229.6/203.5/4.3/4.4, keep the longest Write-tool input or
Read-tool result for the file, and place it here as
RECOVERED__EFFECT_REPORT_old_sglang.md with its transcript provenance.
