"""Small helpers shared by the command-line scripts."""


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
