#!/usr/bin/env python3
"""User-facing wrapper of the offline quantization advisor: setup, plan with a query, choose from the frontier, encode.

`setup` runs the expensive one-time steps (prepare, calibrate) into a work directory. `plan` turns a query (weights,
bounds, budgets) into a configuration change, runs the existing planner on the prepared inputs and shows a thinned
frontier to choose from. All modelling stays in auto_quantize.py and its modules."""
from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sys
from pathlib import Path

from config import AXES, QuantError, capacity, load_config
import frontier as fr

WEIGHT_NAMES = {"loss": "precision", "kld": "precision", "precision": "precision", "decode": "decode", "short": "short",
                "long": "long"}
# Rate bounds of the command line and their configuration names
RATE_BOUNDS = {"min_decode": "min_decode_t_s", "min_short": "min_short_t_s", "min_long": "min_long_t_s"}


def parse_pairs(text, name):
    pairs = {}
    for item in text.split(","):
        key, sep, value = item.partition("=")
        if not sep or not key.strip() or key.strip() in pairs:
            raise QuantError(f"{name} must be KEY=VALUE pairs without repeats: {text!r}")
        pairs[key.strip()] = value.strip()
    return pairs


def parse_weights(text):
    """'loss=.4,decode=.3,short=.2,long=.1' -> importance by axis; omitted axes count 0."""
    result = {axis: 0.0 for axis in AXES}
    for key, value in parse_pairs(text, "--weights").items():
        if key not in WEIGHT_NAMES:
            raise QuantError(f"unknown weight {key!r}; use loss, decode, short, long")
        try:
            result[WEIGHT_NAMES[key]] = float(value)
        except ValueError as exc:
            raise QuantError(f"weight {key} is not a number: {value!r}") from exc
    if any(v < 0 for v in result.values()) or sum(result.values()) <= 0:
        raise QuantError("weights must be nonnegative with at least one positive")
    return result


def apply_query(cfg, args):
    """A copy of the loaded configuration with the query applied. Only planning inputs change, never the prepared ones."""
    cfg = copy.deepcopy(cfg)
    weights = parse_weights(args.weights) if args.weights else None
    if args.minimize in ("kld", "size"):
        if weights:
            raise QuantError(f"--minimize {args.minimize} and --weights are exclusive")
        weights = {axis: float(axis == "precision") for axis in AXES}
    if args.minimize == "size":
        if args.max_kld is None:
            raise QuantError("--minimize size needs --max-kld")
        cfg["search"]["minimize"] = "size"
    if args.max_size is not None:
        cfg["bounds"]["max_total_bytes"] = capacity(args.max_size)
    if weights:
        total = sum(weights.values())
        cfg["importance"] = {axis: v / total for axis, v in weights.items()}
    if args.budget:
        for device, amount in parse_pairs(args.budget, "--budget").items():
            if device not in cfg["devices"]:
                raise QuantError(f"--budget names {device}, which is not a device of the configuration: {sorted(cfg['devices'])}")
            cfg["devices"][device] = capacity(amount)
            if cfg["reserve"].get(device, 0) > cfg["devices"][device]:
                raise QuantError(f"reserve exceeds the {device} budget")
    if args.max_kld is not None:
        if "quality" not in cfg:
            raise QuantError("--max-kld needs a calibrated quality table; run setup with calibration first")
        cfg["bounds"]["max_loss"] = args.max_kld
    for option, name in RATE_BOUNDS.items():
        if getattr(args, option) is not None:
            cfg["bounds"][name] = getattr(args, option)
    return cfg


def paths(work):
    work = Path(work).resolve()
    return {"prepared": work / "prepared", "quality": work / "quality", "plans": work / "plans"}


def command_setup(args):
    import auto_quantize as aq
    from calibrate import calibrate
    cfg = load_config(args.config)
    p = paths(args.work)
    aq.prepare(cfg, p["prepared"], args.measurement_seconds)
    table = p["quality"] / "quality.json"
    if args.skip_calibrate:
        print("Skipped calibration: plans will use the local error proxy, not predicted KL")
    elif table.exists():
        print(f"Reusing {table}")
    else:
        if not cfg["calibration"]["text"]:
            raise QuantError("calibration.text is required (or pass --skip-calibrate)")
        data = aq.load_prepared(p["prepared"] / "prepared.json", cfg)
        calibrate(data, cfg, p["quality"], aq.encode, aq.verify_output, text_dir=p["prepared"])
    print(f"Ready: python {sys.argv[0]} plan --config {args.config} --work {args.work} ...")


def command_plan(args):
    cfg = load_config(args.config)
    p = paths(args.work)
    table = p["quality"] / "quality.json"
    if "quality" not in cfg and table.exists():
        cfg["quality"] = {"table": str(table)}
    cfg = apply_query(cfg, args)
    import auto_quantize as aq
    data = aq.load_prepared(p["prepared"] / "prepared.json", cfg)
    existing = len(list(p["plans"].glob("plan-*"))) if p["plans"].exists() else 0
    out = p["plans"] / (args.name or f"plan-{existing + 1:03d}")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        aq.plan(data, cfg, out)
    lines = log.getvalue().splitlines()
    summary = lines[:1] + [line for line in lines[1:] if line.startswith(("Saved", "Predicted KL"))]
    print("\n".join(summary))
    return choose(json.loads((out / "frontier.json").read_text()), out / "frontier.json", args,
                  default_pick={"kld": "min-kld", "size": "min-size"}.get(args.minimize))


def command_show(args):
    return choose(json.loads(Path(args.frontier).read_text()), Path(args.frontier), args)


def command_encode(args):
    return choose(json.loads(Path(args.frontier).read_text()), Path(args.frontier), args)


def choose(payload, path, args, default_pick=None):
    """Show the thinned frontier, resolve the pick (option, default or prompt) and optionally encode it."""
    all_rows = fr.rows(payload)
    if not all_rows:
        raise QuantError(f"the plan has no feasible recipe ({payload['search']['status']})")
    if payload["search"]["status"] != "ok":
        print(f"Warning: search status {payload['search']['status']}")
    lazy = all_rows[0]["recipe"]["memory"].get("lazy_bytes", 0)
    if lazy:
        print(f"Lazy tables ({lazy / fr.GIB:.2f} GiB) stay on disk and are not counted in sizes or budgets.")
    shown = fr.thin(all_rows, args.rows)
    print(f"{len(all_rows)} nondominated recipes; showing {len(shown)}. * marks the best score; dKL is relative to it.")
    print(fr.table(shown, "pred KL" if payload.get("quality") else "local R"))
    if not payload.get("timing_qualified"):
        print("Speeds are unqualified estimates (run validate to qualify them).")
    spec = getattr(args, "pick", None) or default_pick
    if getattr(args, "detail", None):
        row = fr.pick(shown, all_rows, args.detail)
        for kind, text in fr.detail(row["recipe"]).items():
            print(f"  {kind}: {text}")
    if spec is None and sys.stdin.isatty() and getattr(args, "output", None):
        spec = input("Select a row number, recipe id, best, min-kld, min-size, fastest-decode ... (empty = none): ").strip() or None
    if spec is None:
        return
    row = fr.pick(shown, all_rows, spec)
    print(f"Selected {row['id']}: {row['size'] / fr.GIB:.2f} GiB, KL {row['kld']:.4g}, decode {row['decode']:.2f} t/s")
    if row["recipe"]["memory"].get("runtime_args"):
        print("Run with: " + " ".join(row["recipe"]["memory"]["runtime_args"]))
    if getattr(args, "output", None):
        import auto_quantize as aq
        print(f"Encoding to {args.output}")
        aq.encode_from_frontier(payload, row["id"], args.output)
    else:
        print(f"Encode with: python {sys.argv[0]} encode --frontier {path} --pick {row['id']} --output OUT.gguf")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, work=True):
        if work:
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--work", type=Path, required=True, help="directory for prepared inputs, quality table and plans")

    def display(p):
        p.add_argument("--rows", type=int, default=10, help="rows to show (thinned from the whole frontier)")
        p.add_argument("--pick", help="row number, recipe id, best, min-kld, min-size, fastest-decode|short|long")
        p.add_argument("--detail", help="print the qtype histogram per tensor kind of this pick")
        p.add_argument("--output", type=Path, help="encode the pick to this GGUF (without it the pick is only named)")

    p = sub.add_parser("setup", help="prepare and calibrate once per model and machine state")
    common(p)
    p.add_argument("--measurement-seconds", type=float, default=600)
    p.add_argument("--skip-calibrate", action="store_true")
    p = sub.add_parser("plan", help="plan for a query and choose from the frontier")
    common(p)
    p.add_argument("--weights", help="loss=.4,decode=.3,short=.2,long=.1 (relative; omitted axes count 0)")
    p.add_argument("--minimize", choices=("score", "kld", "size"), default="score",
                   help="score: the weighted objective; kld: precision only; size: smallest recipe with predicted KL <= --max-kld")
    p.add_argument("--budget", help="DEVICE=SIZE,... replaces device budgets for this plan, e.g. ROCm1=24GiB,RAM=400GiB")
    p.add_argument("--max-size", help="ceiling on the summed tensor bytes over all devices, e.g. 55GiB (reserves excluded)")
    p.add_argument("--max-kld", type=float, help="ceiling on the predicted KL (needs a quality table)")
    for option in RATE_BOUNDS:
        p.add_argument("--" + option.replace("_", "-"), dest=option, type=float, help="minimum predicted t/s")
    p.add_argument("--name", help="plan directory name under WORK/plans")
    display(p)
    p = sub.add_parser("show", help="show a saved frontier")
    p.add_argument("--frontier", type=Path, required=True)
    display(p)
    p = sub.add_parser("encode", help="encode a recipe of a saved frontier")
    p.add_argument("--frontier", type=Path, required=True)
    p.add_argument("--rows", type=int, default=10)
    p.add_argument("--pick", required=True)
    p.add_argument("--output", type=Path, required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        {"setup": command_setup, "plan": command_plan, "show": command_show, "encode": command_encode}[args.command](args)
    except (QuantError, ValueError, OSError, KeyError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
