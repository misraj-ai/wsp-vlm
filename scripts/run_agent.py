"""Start or resume one agent run.

    python scripts/run_agent.py --config configs/my_run.yaml
    python scripts/run_agent.py --config configs/my_run.yaml --set budget.passes=10 --set run_dir=runs/claude_b10_s1

Re-running the same command resumes from runs/<name>/state.json after a crash or an API outage.
"""
import argparse

from _common import parse_sets  # noqa: F401  (also puts the repo on sys.path)
from wsp import driver

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--set", action="append", default=[], help="override, e.g. budget.passes=25")
ap.add_argument("--max-turns", type=int)
a = ap.parse_args()
r = driver.main(a.config, parse_sets(a.set), a.max_turns)
print(f"done={r.state['done']} turns={r.state['turn']} passes={r.state['passes']} python_runs={r.state['python_runs']}")
print(f"submission: {r.dir}/submission")
