import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def parse_sets(pairs):
    """--set a.b=value (YAML-parsed value) -> nested dict"""
    import yaml

    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        d = out
        ks = k.split(".")
        for kk in ks[:-1]:
            d = d.setdefault(kk, {})
        d[ks[-1]] = yaml.safe_load(v)
    return out
