#!/usr/bin/env python3
"""User-facing wrapper of the offline quantization advisor: setup, plan with a query, choose from the frontier, encode.

`setup` runs the expensive one-time steps (prepare, calibrate) into a work directory. `plan` turns a query (weights,
bounds, budgets) into a configuration change, runs the existing planner on the prepared inputs and shows a thinned
frontier to choose from. All modelling stays in auto_quantize.py and its modules."""
from __future__ import annotations

import argparse
import contextlib
import copy
import inspect
import io
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

from config import AXES, ROOT, SCHEMA, QuantError, capacity, digest, file_identity, load_config, loss_only, write_json
import capacity as cap
import frontier as fr
import progress
import runstate
import space
import verify as vf

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


def apply_preferences(cfg, args):
    """The weights and speed bounds of the query, applied to cfg in place. They also decide whether prepare measures
    timing: with loss-only weights and no speed bound it does not (config.loss_only)."""
    weights = parse_weights(args.weights) if args.weights else None
    if getattr(args, "minimize", None) in ("kld", "size"):
        if weights:
            raise QuantError(f"--minimize {args.minimize} and --weights are exclusive")
        weights = {axis: float(axis == "precision") for axis in AXES}
    if weights:
        total = sum(weights.values())
        cfg["importance"] = {axis: v / total for axis, v in weights.items()}
    for option, name in RATE_BOUNDS.items():
        if getattr(args, option, None) is not None:
            cfg["bounds"][name] = getattr(args, option)
    return cfg


def apply_query(cfg, args, ratio=None):
    """A copy of the loaded configuration with the query applied. Only planning inputs change, never the prepared ones.

    ratio is the stored measured/predicted KL ratio of an earlier verified encode: --max-kld is divided by it unless
    --kld-correction none is given."""
    cfg = apply_preferences(copy.deepcopy(cfg), args)
    if args.minimize == "size":
        if args.max_kld is None:
            raise QuantError("--minimize size needs --max-kld")
        cfg["search"]["minimize"] = "size"
    if args.max_size is not None:
        cfg["bounds"]["max_total_bytes"] = capacity(args.max_size)
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
        if ratio and getattr(args, "kld_correction", "ratio") != "none":
            cfg["bounds"]["max_loss"] = vf.corrected_ceiling(args.max_kld, ratio)
            print(f"KL correction: measured/predicted ratio {ratio:.3f} from the last verified encode; "
                  f"--max-kld {args.max_kld:g} -> model ceiling {cfg['bounds']['max_loss']:.5g} (--kld-correction none turns it off)")
    return cfg


def paths(work, scratch=None):
    """Directories of a work directory. work keeps the durable results (prepared tables, quality table, plans, state);
    scratch (default WORK/scratch) keeps what can be deleted: calibration probe models, patches and teacher logits.

    A work directory from before the scratch option keeps its calibration files in WORK/quality; they stay there."""
    work = Path(work).resolve()
    quality = work / "quality"
    legacy = scratch is None and (any(quality.glob("session-*")) or (quality / "teacher.kld").exists())
    scratch = Path(scratch).resolve() if scratch else work / "scratch"
    return {"work": work, "scratch": scratch, "prepared": work / "prepared", "quality": quality, "plans": work / "plans",
            "calibration": quality if legacy else scratch / "calibration"}


def directories(p, output=None):
    return {"work": p["work"], "scratch": p["scratch"], "output": Path(output).resolve().parent if output else p["work"]}


def stage_prepare(cfg, p, args, measurement_seconds=600):
    import auto_quantize as aq
    if not (p["prepared"] / "prepared.json").exists():
        space.preflight("prepare", aq.configured_model(cfg), cfg, directories(p), not args.no_space_check)
    return aq.prepare(cfg, p["prepared"], measurement_seconds)


def terminate_children():
    """SIGTERM to the child processes of this process (Linux). calibrate's SIGTERM handler raises SystemExit in the main
    thread, but its session threads are not daemons and keep the process alive until every pass has finished: ending the
    llama-perplexity sessions makes them stop at once."""
    me = os.getpid()
    try:
        entries = [e for e in os.listdir("/proc") if e.isdigit()]
    except OSError:
        return
    for entry in entries:
        try:
            with open(f"/proc/{entry}/stat") as stream:
                if int(stream.read().rsplit(")", 1)[1].split()[1]) == me:
                    os.kill(int(entry), signal.SIGTERM)
        except (OSError, ValueError, IndexError):
            continue


def stage_calibrate(cfg, p, args):
    """Calibration passes with the large files in the scratch directory; only quality.json goes to the work directory."""
    import auto_quantize as aq
    from calibrate import calibrate
    data = aq.load_prepared(p["prepared"] / "prepared.json", cfg)
    info = aq.configured_model(cfg)
    c = cfg["calibration"]
    vocab = space.vocab_size(cfg["model"])
    space.preflight("calibrate", info, cfg, directories(p), not args.no_space_check, vocab=vocab)
    extra = {"scratch": p["scratch"]} if "scratch" in inspect.signature(calibrate).parameters else {}
    # the watcher covers the teacher logits and a probe model; calibrate prints its own pass lines
    try:
        with progress.CalibrationWatcher(p["calibration"], teacher_bytes=space.teacher_bytes(vocab, c["chunks"], c["context"]), passes=False):
            table = calibrate(data, cfg, p["calibration"], aq.encode, aq.verify_output, text_dir=p["prepared"], **extra)
    except (SystemExit, KeyboardInterrupt):
        terminate_children()
        raise
    if p["calibration"] != p["quality"]:
        p["quality"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(p["calibration"] / "quality.json", p["quality"] / "quality.json")
    return table


def command_setup(args):
    cfg = apply_preferences(load_config(args.config), args)
    p = paths(args.work, args.scratch)
    stage_prepare(cfg, p, args, args.measurement_seconds)
    table = p["quality"] / "quality.json"
    if args.skip_calibrate:
        print("Skipped calibration: plans will use the local error proxy, not predicted KL")
    elif table.exists():
        print(f"Reusing {table}")
    else:
        if not cfg["calibration"]["text"]:
            raise QuantError("calibration.text is required (or pass --skip-calibrate)")
        stage_calibrate(cfg, p, args)
    print(f"Ready: python {sys.argv[0]} plan --config {args.config} --work {args.work} ...")


def serve_from_args(args):
    """The serving layout of the options: what the memory reserves and the launched server depend on."""
    return {"ctx": args.ctx, "ub": args.ub, "parallel": args.parallel, "ctk": args.ctk, "ctv": args.ctv,
            "mmproj": str(args.mmproj.resolve()) if args.mmproj else None, "draft": str(args.draft.resolve()) if args.draft else None,
            "draft_device": args.draft_device, "spec_type": args.spec_type, "server_args": args.server_arg or [],
            "mtp": not args.no_mtp}


def measured_capacities(args, bindir, env=None):
    """Capacity table from the memory ggml reports now (explicit sizes in --devices win)."""
    devices = cap.parse_devices(args.devices)
    names = [name for name, _ in devices]
    keep = cap.parse_keep_free(args.keep_free, names)
    ram = cap.host_memory() if "RAM" in names else None
    listed = cap.list_devices(bindir, env) if any(n != "RAM" for n in names) else {}
    return cap.capacity_rows(devices, listed, keep, args.capacity, ram)


def command_capacity(args):
    """Print capacities and the measured reserves of a model and serving layout (no weights are loaded)."""
    import model as mdl
    bindir = (args.bindir or ROOT / "build-auto-quantizer/bin").resolve()
    rows = measured_capacities(args, bindir)
    print(cap.table_text(rows))
    gpus = [r["device"] for r in rows if r["device"] != "RAM"]
    info = mdl.read_model(str(args.model))
    split = cap.initial_split(info["n_layer"], [r["capacity"] for r in rows if r["device"] != "RAM"])
    cfg = {"bindir": str(bindir), "model": str(args.model.resolve()), "runtime": {"env": {}}}
    measured = cap.measure_reserves(cfg, serve_from_args(args), gpus, split, "RAM" in [r["device"] for r in rows])
    print(f"Split by capacity (-ts): {','.join(map(str, split))}; command: {' '.join(measured['command'])}")
    print(cap.reserves_text(measured, split))


def work_of(frontier_path):
    """WORK of WORK/plans/plan-N/frontier.json, else None."""
    plan = Path(frontier_path).resolve().parent
    return plan.parent.parent if plan.parent.name == "plans" else None


def stage_verify(payload, row, output, args, work=None, scratch=None):
    """Check an encoded model: server load with the planned layout and free memory per GPU, one request, then the KL
    against the quality table's teacher (measured against predicted; the ratio goes to the work directory)."""
    import auto_quantize as aq
    cfg = payload["config"]
    output = Path(output).resolve()
    scratch = Path(scratch) if scratch else (work / "scratch" if work else output.parent) / "verify"
    names = [n for n in cfg["devices"] if n != "RAM"]
    keep = cap.parse_keep_free(args.keep_free, names)
    rows = [{"device": n, "keep_free": keep[n]} for n in names]
    report = {"output": str(output), "recipe": row["id"]}
    server = vf.run_server_check(cfg, output, row["recipe"]["memory"].get("runtime_args"), serve_from_args(args), rows,
                                 scratch, args.verify_timeout)
    report["server"] = server
    print(f"Server loaded in {server['load_seconds']:.0f} s and answered: {server['answer'].strip()[:80]!r}")
    if server.get("long"):
        print(f"A {server['long']['tokens']}-token prompt prefilled at {server['long']['prefill_t_s'] or 0:.0f} t/s")
    print(vf.memory_text(server["memory"]))
    runtime = vf.measured_runtime(cfg, row, keep, server["memory"])
    if runtime and work:
        write_json(work / cap.RUNTIME_FILE, {"mtp": cfg.get("mtp", False), "ub": serve_from_args(args)["ub"], "per_device": runtime})
        print("Memory outside ggml buffers per GPU (planned -> measured): "
              + ", ".join(f"{n} {cfg.get('runtime_reserve', {}).get(n, 0) // cap.MIB} -> {b // cap.MIB} MiB" for n, b in runtime.items())
              + f"; stored in {work / cap.RUNTIME_FILE} for the next plans")
    if not all(m["ok"] for m in server["memory"]):
        print("Warning: a GPU is below its keep-free amount. Run again: the reserves use the measured runtime memory "
              "now, and plan and encode run again (or use a larger --keep-free or a smaller --ctx)")
    if "quality" in cfg:
        table = json.loads(Path(cfg["quality"]["table"]).read_text())
        table = vf.ensure_teacher(cfg, table, scratch)
        measured, se = aq.witness_loss(cfg, table, output, scratch, "verify")
        ratio = vf.kl_ratio(measured, row["kld"])
        report["kl"] = {"measured": measured, "se": se, "predicted": row["kld"], "ratio": ratio}
        print(f"KL measured {measured:.5f} +- {se:.5f}, predicted {row['kld']:.5f}, ratio measured/predicted {ratio:.3f}")
        if work:
            vf.store_ratio(work, {"ratio": ratio, **report["kl"], "recipe": row["id"], "output": str(output)})
            print(f"Stored the ratio in {work / vf.RATIO_FILE}; plan --max-kld divides its ceiling by it")
    if work:
        write_json(work / "verify.json", report)
    return report


def run_plan(cfg, p, out):
    """Plan into `out` on the prepared inputs (quiet), print the summary lines and return the frontier payload."""
    import auto_quantize as aq
    data = aq.load_prepared(p["prepared"] / "prepared.json", cfg)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        aq.plan(data, cfg, out)
    lines = log.getvalue().splitlines()
    print("\n".join(lines[:1] + [line for line in lines[1:] if line.startswith(("Saved", "Predicted KL"))]))
    return json.loads((Path(out) / "frontier.json").read_text())


def command_plan(args):
    cfg = load_config(args.config)
    p = paths(args.work, args.scratch)
    table = p["quality"] / "quality.json"
    if "quality" not in cfg and table.exists():
        cfg["quality"] = {"table": str(table)}
    cfg = apply_query(cfg, args, vf.load_ratio(p["work"]))
    existing = len(list(p["plans"].glob("plan-*"))) if p["plans"].exists() else 0
    out = p["plans"] / (args.name or f"plan-{existing + 1:03d}")
    payload = run_plan(cfg, p, out)
    return choose(payload, out / "frontier.json", args, default_pick={"kld": "min-kld", "size": "min-size"}.get(args.minimize))


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
    if not payload.get("timed", True):
        print("Speed was not measured (loss-only weights).")
    elif not payload.get("timing_qualified"):
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
    print(f"Selected {row['id']}: {fr.selected_text(row)}")
    if row["recipe"]["memory"].get("runtime_args"):
        print("Run with: " + " ".join(row["recipe"]["memory"]["runtime_args"]))
    if getattr(args, "output", None):
        import auto_quantize as aq
        out = Path(args.output).resolve().parent
        info = payload["prepared"]["model"]
        space.check("encode", [("encoded GGUF", "output", space.recipe_bytes(info, row["recipe"]["tensor_types"]))],
                    {"output": out}, not args.no_space_check)
        print(f"Encoding to {args.output}")
        aq.encode_from_frontier(payload, row["id"], args.output)
        if not args.no_verify:
            work = work_of(path)
            stage_verify(payload, row, args.output, args, work, getattr(args, "scratch", None))
    else:
        print(f"Encode with: python {sys.argv[0]} encode --frontier {path} --pick {row['id']} --output OUT.gguf")


ROUND = 64 << 20


def rounded(values):
    """Bytes per device rounded down to 64 MiB, so that the free memory drifting by a few MiB does not re-run the plan."""
    return {k: v // ROUND * ROUND for k, v in values.items()}


def absolutize_override(raw, base):
    """Paths of a user --config that load_config would read relative to the work directory, made relative to the file."""
    def absolute(value):
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (Path(base) / path).resolve())
    raw = copy.deepcopy(raw)
    if "bindir" in raw:
        raw["bindir"] = absolute(raw["bindir"])
    for key in ("text", "teacher"):
        value = raw.get("calibration", {}).get(key)
        if isinstance(value, str) and "," not in value and ":" not in value:
            raw["calibration"][key] = absolute(value)
    return raw


def build_config(args, gguf, rows, user_reserve, measured, override):
    """The advisor configuration of a run: the options, the measured capacities and reserves, and the user's --config.

    An explicit device size of the config wins over the measured capacity; its reserve adds to the measured one."""
    from config import devices as parse_devices
    names = [r["device"] for r in rows]
    capacities = {r["device"]: r["capacity"] for r in rows}
    for name, amount in parse_devices(override["devices"]).items() if "devices" in override else ():
        if name not in capacities:
            raise QuantError(f"the config names device {name}, which is not in --devices {','.join(names)}")
        capacities[name] = amount
    reserve = {name: e["total"] + user_reserve.get(name, 0) for name, e in measured["per_device"].items()}
    runtime = {name: e.get("runtime", 0) for name, e in measured["per_device"].items()}
    for name in user_reserve:
        if name not in capacities:
            raise QuantError(f"the config reserve names {name}, which is not in --devices")
    cfg = {"schema": SCHEMA, "model": str(gguf), "devices": capacities, "reserve": reserve, "imatrix": [str(x.resolve()) for x in args.imatrix],
           # the converted source is Q8_0; with an F16/BF16 source the option changes nothing
           "allow_requantize": True, "mtp": measured.get("mtp", False), "runtime_reserve": runtime}
    if args.bindir:
        cfg["bindir"] = str(args.bindir.resolve())
    if args.calibration_text:
        texts = [str(Path(path).resolve()) + sep + weight for path, sep, weight in (t.partition(":") for t in args.calibration_text)]
        cfg["calibration"] = {"text": texts if len(texts) > 1 else texts[0]}
    cfg["runtime"] = {}
    if args.ub != 512:
        cfg["runtime"].update(ubatch=args.ub, batch=max(2048, args.ub))
    cfg["runtime"]["extra_args"] = [x for flag, v in (("-ctk", args.ctk), ("-ctv", args.ctv)) if v for x in (flag, v)]
    for key, value in override.items():
        if key in ("devices", "reserve"):
            continue
        if key in ("schema", "model"):
            raise QuantError(f"the config must not set {key}: the run decides it")
        if isinstance(value, dict) and isinstance(cfg.get(key), dict):
            merged = {**cfg[key], **value}
            if key == "runtime":
                merged["extra_args"] = cfg[key]["extra_args"] + list(value.get("extra_args", []))
            cfg[key] = merged
        else:
            cfg[key] = value
    return cfg


# The prepare inputs that change what a calibration pass measures. The others (runtime, timing, machine settings, the
# expert cache) only change the timing tables.
CALIBRATION_PREPARE_KEYS = ("model", "imatrix", "sampling", "groups", "binaries", "libraries", "gpu", "ram_routed",
                            "layer_split_gpus")


def prepare_stage_inputs(state, cfg, inputs):
    """The inputs of the prepare stage. With loss-only weights a finished prepare with timing serves as it is; otherwise the
    key loss_only marks a prepare without timing, so a later run with a time importance or a speed bound runs prepare
    again and measures the timing. Calibration does not follow the key (CALIBRATION_PREPARE_KEYS)."""
    if loss_only(cfg):
        recorded = state.stages.get("prepare", {})
        if not ("finished" in recorded and recorded.get("inputs") == inputs):
            return {**inputs, "loss_only": True}
    return inputs


def calibration_previous(prepare_inputs, cfg, converted):
    """Fingerprint that the calibrate stage chains on, in place of the one of prepare. The libraries are identified by
    their files, so LD_LIBRARY_PATH is left out of the inherited environment."""
    keys = {k: prepare_inputs[k] for k in CALIBRATION_PREPARE_KEYS if k in prepare_inputs}
    keys["environment"] = digest({"inherited": {name: value for name, value in os.environ.items()
                                                if name.startswith(("GGML_", "LLAMA_", "CUDA_", "HIP_", "HSA_", "ROCR_", "OMP_"))},
                                  "runtime": cfg["runtime"].get("env", {})})
    return runstate.fingerprint(keys, converted)


def keep_calibration(state, inputs, previous, p, foreign):
    """--keep-calibration: record the finished calibration as one of the current inputs, for a rebuild of the
    binaries that keeps the numbers. Its own inputs (calibration settings and text) and its model must be the same.
    When a run without the option already moved it to SCRATCH/calibration.replaced, it is moved back."""
    table = p["quality"] / "quality.json"
    record = state.stages.get("calibrate")
    if not (record and "finished" in record and record.get("inputs") == inputs and table.exists()):
        record = restore_replaced_calibration(inputs, p, foreign)
    if record is None:
        print("--keep-calibration: no finished calibration with the same settings; calibrate runs as usual", flush=True)
        return
    if (why := foreign()) is not None:
        print(f"--keep-calibration: {why}; calibrate runs as usual", flush=True)
        return
    if record.get("fingerprint") != runstate.fingerprint(inputs, previous):
        print("Stage calibrate: keeping the finished calibration although the binaries, libraries or environment "
              "changed (--keep-calibration)", flush=True)
    state.mark("calibrate", inputs, previous, record.get("seconds", 0), record.get("outputs", [str(table)]))


def restore_replaced_calibration(inputs, p, foreign):
    """The record of a finished calibration in SCRATCH/calibration.replaced with the same inputs and model, after its
    files are moved back to SCRATCH/calibration (the files there go to calibration.replaced). None if there is none."""
    kept = p["calibration"].with_name(p["calibration"].name + ".replaced")
    saved = kept / "record.json"
    if not (saved.exists() and (kept / "quality.json").exists()):
        return None
    record = json.loads(saved.read_text())
    if "finished" not in record or record.get("inputs") != inputs or foreign(kept) is not None:
        return None
    swap = p["calibration"].with_name(p["calibration"].name + ".swap")
    shutil.rmtree(swap, ignore_errors=True)
    if p["calibration"].exists():
        p["calibration"].rename(swap)
    kept.rename(p["calibration"])
    if swap.exists():
        swap.rename(kept)
    (p["calibration"] / "record.json").unlink()
    if p["calibration"] != p["quality"]:
        p["quality"].mkdir(parents=True, exist_ok=True)
        shutil.copy2(p["calibration"] / "quality.json", p["quality"] / "quality.json")
    print(f"--keep-calibration: moved the finished calibration from {kept} back to {p['calibration']}"
          + (f"; the files of the later run are now in {kept}" if kept.exists() else ""), flush=True)
    return record


def split_changed(row, used):
    split = row["recipe"]["memory"].get("layer_split")
    return bool(split) and split != used


def check_reserves(cfg, payload, row, serve, used_split, user_reserve, replan, spec, work=None):
    """After the pick: measure the reserves of the chosen layer split. When they differ from the ones planned with, check
    that the recipe still fits, and re-plan once with the new reserves when it does not. Returns (payload, row, report)."""
    from search import memory
    info = payload["prepared"]["model"]
    gpus = [n for n in cfg["devices"] if n != "RAM"]
    split = row["recipe"]["memory"].get("layer_split")
    if not split_changed(row, used_split):
        return payload, row, {"split": used_split, "remeasured": False}
    measured = cap.measure_reserves(cfg, serve, gpus, split, "RAM" in cfg["devices"], work)
    fresh = {n: e["total"] + user_reserve.get(n, 0) for n, e in measured["per_device"].items()}
    print(f"Layer split of the pick {split} differs from {used_split} used for the reserves; measured again:")
    print(cap.reserves_text(measured, split))
    if all(abs(fresh[n] - cfg["reserve"][n]) < ROUND for n in fresh):
        print("Reserves unchanged within 64 MiB.")
        return payload, row, {"split": split, "remeasured": True, "replanned": False, "reserves": fresh}
    trial = copy.deepcopy(cfg)
    trial["reserve"] = fresh
    headroom = memory(info, trial, row["recipe"]["choices"])["headroom_bytes"]
    low = min(headroom[n] for n in gpus)
    if low >= 0:
        print(f"Check passed: the pick fits with the reserves of its split (least headroom {low / fr.GIB:.2f} GiB).")
        return payload, row, {"split": split, "remeasured": True, "replanned": False, "reserves": fresh}
    print(f"The pick misses its budget by {-low / fr.GIB:.2f} GiB with the reserves of its split; planning once more with them.")
    payload = replan(trial)
    rows = fr.rows(payload)
    if not rows:
        raise QuantError("no recipe fits with the reserves of the chosen layer split")
    row = fr.pick(fr.diverse(rows, 15), rows, spec if spec != "ASK" else "best")
    final = memory(info, trial, row["recipe"]["choices"])["headroom_bytes"]
    if min(final[n] for n in gpus) < 0:
        raise QuantError("the re-planned recipe still misses a budget; use a larger --keep-free or a smaller --ctx")
    return payload, row, {"split": row["recipe"]["memory"].get("layer_split"), "remeasured": True, "replanned": True, "reserves": fresh}


def pick_row(payload, args, default_pick):
    """The row to encode: --pick, --auto, or a choice from the diverse table on a terminal (no terminal: --auto with a note)."""
    all_rows = fr.rows(payload)
    if not all_rows:
        raise QuantError(f"the plan has no feasible recipe ({payload['search']['status']})")
    shown = fr.diverse(all_rows, args.rows)
    label = "pred KL" if payload.get("quality") else "local R"
    print(f"{len(all_rows)} nondominated recipes; showing {len(shown)} (best score marked *).")
    print(fr.run_table(shown, label))
    if not payload.get("timed", True):
        print("Speed was not measured (loss-only weights).")
    elif not payload.get("timing_qualified"):
        print("Speeds are unqualified estimates.")
    if args.pick:
        return fr.pick(shown, all_rows, args.pick)
    if args.auto:
        return fr.pick(shown, all_rows, default_pick or "best")
    if not sys.stdin.isatty():
        print(f"No terminal: taking {default_pick or 'best'} as with --auto.")
        return fr.pick(shown, all_rows, default_pick or "best")
    while True:
        answer = input("Select a row number or recipe id (best, min-kld, min-size, fastest-decode ...; empty = best): ").strip()
        try:
            return fr.pick(shown, all_rows, answer or "best")
        except ValueError as exc:
            print(exc)


def run_stage(state, name, inputs, previous, action, outputs=(), reset=None, foreign=None):
    """One resumable stage: skip it when finished, otherwise say why it runs, run it and record it.

    Returns (fingerprint, record). A stage that re-runs because its inputs changed also re-runs the later stages and
    calls reset to remove its files from the earlier inputs. foreign() names such files (or returns None) when the
    state has no record of the stage, for example in a work directory of another model before the stage records."""
    decision, why = state.decide(name, inputs, previous)
    fp = runstate.fingerprint(inputs, previous)
    if decision == "skip":
        print(f"Stage {name}: finished earlier with the same inputs, skipping", flush=True)
        state.skipped.add(name)
        return fp, state.stages[name]
    if decision == "run" and not why and foreign and (found := foreign()):
        decision, why = "rerun", [found]
    if decision == "rerun":
        print(f"Stage {name}: input changed ({', '.join(why)}); running it again and the stages after it", flush=True)
        state.forget_from(name)
        if reset:
            reset()
    elif decision == "resume":
        print(f"Stage {name}: continuing an interrupted run with the same inputs", flush=True)
    elif why:
        print(f"Stage {name}: {why[0]}; running it again", flush=True)
    state.start(name, inputs, previous)
    started = time.monotonic()
    print(f"Stage {name}: running", flush=True)
    extra = action() or {}
    state.mark(name, inputs, previous, time.monotonic() - started, outputs() if callable(outputs) else outputs, **extra)
    return fp, state.stages[name]


def same_tensors(gguf, info):
    """Whether a GGUF has the tensor names of the model (a cheap check that reads only the header)."""
    import model as mdl
    try:
        return {t["name"] for t in mdl.read_model(gguf)["tensors"]} == {t["name"] for t in info["tensors"]}
    except (OSError, ValueError, KeyError):
        return False


def command_run(args):
    import auto_quantize as aq
    import convert as cv
    import model as mdl
    if not 5 <= args.rows <= 15:
        raise QuantError("--rows must be 5 to 15 for run")
    if args.auto and args.pick:
        raise QuantError("--auto and --pick are exclusive")
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    p = paths(work, args.scratch)
    state = runstate.State(work)
    enabled = not args.no_space_check
    override = absolutize_override(json.loads(args.config.read_text()), args.config.resolve().parent) if args.config else {}
    bindir = (args.bindir or Path(override.get("bindir", ROOT / "build-auto-quantizer/bin"))).resolve()
    args.bindir = bindir
    timings = {}
    t_run = time.monotonic()

    def timed(name, fp_record):
        timings[name] = fp_record[1].get("seconds", 0.0)
        return fp_record[0]

    # 1. source GGUF
    source = args.source.resolve()
    inputs = {**cv.source_identity(source), "mmproj": str(args.mmproj.resolve()) if args.mmproj else None}

    def convert_action():
        if cv.is_gguf(source):
            return {"gguf": str(source), "mmproj": str(args.mmproj.resolve()) if args.mmproj else None}
        recorded = state.stages.get("convert", {}).get("gguf_identity")
        gguf, projector = cv.convert(source, p["scratch"] / "convert", args.converter_python, args.mmproj,
                                     check=lambda rows: space.check("convert", rows, directories(p), enabled))
        st = gguf.stat()
        if recorded and recorded["bytes"] == st.st_size:
            # the same conversion again: keep the file identity, so that the prepared inputs stay valid
            os.utime(gguf, ns=(st.st_atime_ns, recorded["mtime_ns"]))
        st = gguf.stat()
        return {"gguf": str(gguf), "mmproj": str(projector) if projector else None,
                "gguf_identity": {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns}}

    def convert_outputs():
        r = state.stages.get("convert", {})
        return [x for x in (r.get("gguf") if not cv.is_gguf(source) else None,) if x]
    fp = timed("convert", run_stage(state, "convert", inputs, None, convert_action, convert_outputs))
    gguf, mmproj = state.stages["convert"]["gguf"], state.stages["convert"]["mmproj"]
    if mmproj:
        args.mmproj = Path(mmproj)

    # 2. capacities and reserves (measured at every start)
    rows = measured_capacities(args, bindir, override.get("runtime", {}).get("env"))
    print("Device capacities (free memory reported by ggml at start, minus keep-free):")
    print(cap.table_text(rows))
    gpus = [r["device"] for r in rows if r["device"] != "RAM"]
    info_source = mdl.read_model(gguf)
    split0 = cap.initial_split(info_source["n_layer"], [r["capacity"] for r in rows if r["device"] != "RAM"])
    from config import devices as parse_devices, capacity as parse_capacity
    user_reserve = {k: parse_capacity(v) for k, v in override.get("reserve", {}).items()}
    pre = {"bindir": str(bindir), "model": gguf, "runtime": {"env": override.get("runtime", {}).get("env", {})}}
    serve = serve_from_args(args)
    with progress.stage("reserves: llama-fit-params"):
        measured = cap.measure_reserves(pre, serve, gpus, split0, "RAM" in [r["device"] for r in rows], work)
    print(f"Reserves measured for the layer split {split0} at -c {args.ctx} (KV and recurrent state, compute buffers, projector, "
          f"{'MTP draft context' if measured['mtp'] else 'draft'}, runtime memory outside ggml buffers):")
    print(cap.reserves_text(measured, split0))
    cfg_raw = build_config(args, gguf, rows, user_reserve, measured, override)
    for name, c in cfg_raw["devices"].items():
        if name != "RAM" and cfg_raw["reserve"][name] >= c:
            raise QuantError(f"{name}: the reserve {cfg_raw['reserve'][name] / fr.GIB:.2f} GiB is not below the capacity {c / fr.GIB:.2f} GiB; "
                             "lower --ctx, add devices or lower --keep-free")
    write_json(work / "config.json", cfg_raw)
    write_json(work / "reserves.json", {"split": split0, "command": measured["command"], "per_device": measured["per_device"],
                                         "capacities": {r["device"]: r["capacity"] for r in rows}})
    cfg = load_config(work / "config.json")
    if not cfg["imatrix"]:
        raise QuantError("run needs --imatrix: without activation statistics the local error misranks groups")
    if not cfg["calibration"]["text"]:
        raise QuantError("run needs --calibration-text (or calibration.text in --config)")
    # the weights and speed bounds decide whether prepare measures timing
    apply_preferences(cfg, args)

    # 3. prepare
    info = aq.configured_model(cfg)
    prepare_inputs = prepare_stage_inputs(state, cfg, {k: digest(v) for k, v in aq.inputs(cfg, info).items()})
    converted = fp

    def prepare_action():
        stage_prepare(cfg, p, args, args.measurement_seconds)

    def prepare_reset():
        (p["prepared"] / "prepared.json").unlink(missing_ok=True)
        baseline = p["prepared"] / "baseline.gguf"
        if baseline.exists() and not same_tensors(baseline, info):
            print(f"Removing {baseline}: it is a baseline of another model", flush=True)
            baseline.unlink()

    def prepare_foreign():
        snapshot = p["prepared"] / "prepared.json"
        if snapshot.exists() and json.loads(snapshot.read_text()).get("input_key") != digest(aq.inputs(cfg, info)):
            return f"{snapshot} is for other inputs"
        return None
    fp = timed("prepare", run_stage(state, "prepare", prepare_inputs, fp, prepare_action, [p["prepared"] / "prepared.json"],
                                    prepare_reset, prepare_foreign))

    # 4. calibrate
    calibration_inputs = {"calibration": cfg["calibration"],
                          "text": [file_identity(e["path"]) for e in (cfg["calibration"]["text"] if isinstance(cfg["calibration"]["text"], list)
                                                                    else [{"path": cfg["calibration"]["text"]}])]}

    def calibrate_reset():
        (p["quality"] / "quality.json").unlink(missing_ok=True)
        # The passes take hours (GLM 5.3: 16 h), so the files of the earlier inputs stay beside the new ones until the
        # next reset: a mistaken option or an edited state.json does not lose them.
        if p["calibration"].exists():
            kept = p["calibration"].with_name(p["calibration"].name + ".replaced")
            shutil.rmtree(kept, ignore_errors=True)
            p["calibration"].rename(kept)
            if calibrate_before and "finished" in calibrate_before:
                write_json(kept / "record.json", calibrate_before)  # for --keep-calibration
            print(f"Moved the calibration files of the earlier inputs to {kept}", flush=True)
            p["quality"].mkdir(parents=True, exist_ok=True)
        # the measured/predicted KL ratio belongs to the old quality table
        (work / vf.RATIO_FILE).unlink(missing_ok=True)
    def calibrate_foreign(directory=p["calibration"]):
        record = directory / "teacher.json"
        model = json.loads(record.read_text()).get("model") if record.exists() else None
        return f"{directory} holds teacher logits of another model" if model not in (None, file_identity(cfg["model"])) else None
    # The calibration passes measure KL, so they follow only the prepare inputs that change the probe models or their
    # numbers: a new -ub, timing setting or loader path runs prepare again but keeps the passes (GLM 5.3: 16 h).
    prepared = fp
    calibrated_on = calibration_previous(prepare_inputs, cfg, converted)
    if args.keep_calibration:
        keep_calibration(state, calibration_inputs, calibrated_on, p, calibrate_foreign)
    calibrate_before = state.stages.get("calibrate")
    fp = timed("calibrate", run_stage(state, "calibrate", calibration_inputs, calibrated_on,
                                      lambda: stage_calibrate(cfg, p, args) and None,
                                      [p["quality"] / "quality.json"], calibrate_reset, calibrate_foreign))
    fp = runstate.fingerprint({"prepare": prepared}, fp)
    cfg["quality"] = {"table": str(p["quality"] / "quality.json")}

    # 5. plan and pick (with the check of the reserves of the chosen split)
    ratio = vf.load_ratio(work)
    cfg = apply_query(cfg, args, ratio)
    default_pick = {"kld": "min-kld", "size": "min-size"}.get(args.minimize)
    plan_inputs = {"importance": cfg["importance"], "bounds": cfg["bounds"], "search": cfg["search"], "mtp": cfg["mtp"],
                   "devices": rounded(cfg["devices"]), "reserve": rounded(cfg["reserve"]), "kld_ratio": ratio,
                   "pick": args.pick or ("auto" if args.auto else None)}
    out = p["plans"] / "run"
    state_box = {}

    def plan_action():
        shutil.rmtree(out, ignore_errors=True)
        payload = run_plan(cfg, p, out)
        row = pick_row(payload, args, default_pick)
        print(f"Selected {row['id']}: {fr.selected_text(row)}")

        def replan(trial):
            shutil.rmtree(out, ignore_errors=True)
            return run_plan(trial, p, out)
        payload, row, report = check_reserves(cfg, payload, row, serve, split0, user_reserve, replan, args.pick or default_pick or "best",
                                              work)
        write_json(work / "reserves.json", {**json.loads((work / "reserves.json").read_text()), "chosen": report})
        state_box.update(payload=payload)
        return {"pick": row["id"]}
    fp = timed("plan", run_stage(state, "plan", plan_inputs, fp, plan_action, [out / "frontier.json"], lambda: shutil.rmtree(out, ignore_errors=True)))
    payload = json.loads((out / "frontier.json").read_text())
    row = next(r for r in fr.rows(payload) if r["id"] == state.stages["plan"]["pick"])
    if row["recipe"]["memory"].get("runtime_args"):
        print("Planned layout: " + " ".join(row["recipe"]["memory"]["runtime_args"]))

    # 6. encode
    output = args.output.resolve()
    encode_inputs = {"output": str(output), "recipe": digest(row["recipe"]["tensor_types"])}

    def encode_action():
        partial = output.with_suffix(".partial" + output.suffix)
        if partial.exists():
            print(f"Removing the partial output of an interrupted encode: {partial}", flush=True)
            partial.unlink()
        space.check("encode", [("encoded GGUF", "output", space.recipe_bytes(payload["prepared"]["model"], row["recipe"]["tensor_types"]))],
                    {"output": output.parent}, enabled)
        aq.encode_from_frontier(payload, row["id"], output)
    decision = state.decide("encode", encode_inputs, fp)[0]
    if decision != "skip" and output.exists():
        raise QuantError(f"output already exists and was not written by this plan: {output}; remove it or use another --output")
    fp = timed("encode", run_stage(state, "encode", encode_inputs, fp, encode_action, [output]))

    # 7. verify
    if not args.no_verify:
        verify_inputs = {"serve": serve, "keep_free": args.keep_free, "output": file_identity(output)}
        fp = timed("verify", run_stage(state, "verify", verify_inputs, fp,
                                       lambda: stage_verify(payload, row, output, args, work, p["scratch"] / "verify") and None))
        report = json.loads((work / "verify.json").read_text()) if (work / "verify.json").exists() else {}
        if report:
            print("Server command: " + report["server"]["command"])
    print("Stage times: " + ", ".join(f"{k} {progress.duration_text(v)}" + (" (earlier run)" if k in state.skipped else "")
                                          for k, v in timings.items())
          + f"; this run {progress.duration_text(time.monotonic() - t_run)}")
    if not args.keep_scratch and p["scratch"].exists():
        freed = sum(f.stat().st_size for f in p["scratch"].rglob("*") if f.is_file())
        shutil.rmtree(p["scratch"], ignore_errors=True)
        print(f"Removed the scratch directory {p['scratch']} ({progress.size_text(freed)}); --keep-scratch keeps it")
    print(f"Done: {output}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, work=True):
        if work:
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--work", type=Path, required=True, help="directory for prepared inputs, quality table and plans")

    def space_options(p, scratch=True):
        if scratch:
            p.add_argument("--scratch", type=Path, help="directory for temporary files (probe models, patches, teacher logits); "
                           "default WORK/scratch")
        p.add_argument("--no-space-check", action="store_true", help="skip the free disk space check before large writes")

    def display(p):
        p.add_argument("--rows", type=int, default=10, help="rows to show (thinned from the whole frontier)")
        p.add_argument("--pick", help="row number, recipe id, best, min-kld, min-size, fastest-decode|short|long")
        p.add_argument("--detail", help="print the qtype histogram per tensor kind of this pick")
        p.add_argument("--output", type=Path, help="encode the pick to this GGUF (without it the pick is only named)")
        space_options(p)
        verify_options(p)

    def serving_options(p, required=True):
        p.add_argument("--ctx", type=int, required=required, default=None if required else 8192,
                       help="context size to serve (sets the KV reserve)")
        p.add_argument("--mmproj", type=Path, help="multimodal projector file (its memory is measured)")
        p.add_argument("--draft", type=Path, help="draft model or MTP head file (counts its file size)")
        p.add_argument("--draft-device", help="device of the draft file (default: the first GPU)")
        p.add_argument("--spec-type", default="draft-mtp", help="llama-server --spec-type for --draft (default draft-mtp)")
        p.add_argument("--no-mtp", action="store_true",
                       help="serve without the MTP blocks of the model (default: they draft on the GPU, and the plan and "
                            "verify count and run them)")
        p.add_argument("-ctk", "--cache-type-k", dest="ctk", help="K cache type (default f16)")
        p.add_argument("-ctv", "--cache-type-v", dest="ctv", help="V cache type (default f16)")
        p.add_argument("-ub", "--ubatch-size", dest="ub", type=int, default=512)
        p.add_argument("--parallel", type=int, default=1)
        p.add_argument("--server-arg", action="append", help="one more llama-server argument (repeat per word)")

    def capacity_options(p):
        p.add_argument("--devices", required=True, help="GPUs in layer order, optional sizes: ROCm3,ROCm0,ROCm2=24GiB,RAM=200GiB")
        p.add_argument("--keep-free", help="memory left free per GPU: SIZE or ROCm2=2GiB,ROCm0=1GiB (default 512MiB)")
        p.add_argument("--capacity", choices=("free", "total"), default="free",
                       help="plan on the memory free now (default) or on the total memory")
        p.add_argument("--bindir", type=Path, help="build bin directory (default build-auto-quantizer/bin of the repository)")

    def verify_options(p):
        serving_options(p, required=False)
        p.add_argument("--keep-free", help="verify: memory each GPU must keep free: SIZE or ROCm2=2GiB,... (default 512MiB)")
        p.add_argument("--no-verify", action="store_true", help="do not load the encoded model in llama-server and measure its KL")
        p.add_argument("--verify-timeout", type=float, default=3600, help="seconds to wait for the server to load")

    p = sub.add_parser("capacity", help="print device capacities and the measured reserves of a model and context")
    p.add_argument("--model", type=Path, required=True, help="GGUF of the model (the source or any quant of it)")
    capacity_options(p)
    serving_options(p)
    p = sub.add_parser("setup", help="prepare and calibrate once per model and machine state")
    common(p)
    p.add_argument("--weights", help="as for plan; loss=1 alone skips the timing measurement (plans then weigh only the loss)")
    p.add_argument("--measurement-seconds", type=float, default=600)
    p.add_argument("--skip-calibrate", action="store_true")
    space_options(p)
    p = sub.add_parser("plan", help="plan for a query and choose from the frontier")
    common(p)
    p.add_argument("--weights", help="loss=.4,decode=.3,short=.2,long=.1 (relative; omitted axes count 0)")
    p.add_argument("--minimize", choices=("score", "kld", "size"), default="score",
                   help="score: the weighted objective; kld: precision only; size: smallest recipe with predicted KL <= --max-kld")
    p.add_argument("--budget", help="DEVICE=SIZE,... replaces device budgets for this plan, e.g. ROCm1=24GiB,RAM=400GiB")
    p.add_argument("--max-size", help="ceiling on the summed tensor bytes over all devices, e.g. 55GiB (reserves excluded)")
    p.add_argument("--max-kld", type=float, help="ceiling on the KL (needs a quality table); divided by the measured/predicted "
                   "ratio of the last verified encode when the work directory has one")
    p.add_argument("--kld-correction", choices=("ratio", "none"), default="ratio",
                   help="none: use --max-kld as the ceiling on the predicted KL, without the stored ratio")
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
    space_options(p, scratch=False)
    verify_options(p)
    p = sub.add_parser("run", help="convert, measure, prepare, calibrate, plan, encode and verify in one command (resumable)")
    p.add_argument("--source", type=Path, required=True, help="safetensors directory (converted to Q8_0) or a GGUF")
    p.add_argument("--work", type=Path, required=True, help="directory for results and the resume state")
    p.add_argument("--config", type=Path, help="JSON that overrides fields of the generated configuration (qtype pools, runtime, "
                   "calibration, explicit device sizes, extra reserve)")
    p.add_argument("--imatrix", type=Path, action="append", default=[], help="activation statistics file (repeat for several)")
    p.add_argument("--calibration-text", action="append", default=[], help="calibration text PATH or PATH:WEIGHT (repeat)")
    capacity_options(p)
    serving_options(p)
    p.add_argument("--weights", help="loss=.4,decode=.3,short=.2,long=.1 (relative; omitted axes count 0)")
    p.add_argument("--minimize", choices=("score", "kld", "size"), default="score")
    p.add_argument("--max-size", help="ceiling on the summed tensor bytes over all devices, e.g. 55GiB")
    p.add_argument("--max-kld", type=float, help="ceiling on the KL; divided by the stored measured/predicted ratio")
    p.add_argument("--kld-correction", choices=("ratio", "none"), default="ratio")
    p.add_argument("--budget", help=argparse.SUPPRESS)
    for option in RATE_BOUNDS:
        p.add_argument("--" + option.replace("_", "-"), dest=option, type=float, help="minimum predicted t/s")
    p.add_argument("--output", type=Path, required=True, help="encoded GGUF")
    p.add_argument("--auto", action="store_true", help="encode the best pick without asking")
    p.add_argument("--pick", help="encode this row number or recipe id (or best, min-kld ...) without asking")
    p.add_argument("--rows", type=int, default=10, help="rows of the choice table, 5 to 15")
    p.add_argument("--scratch", type=Path, help="directory for temporary files; default WORK/scratch")
    p.add_argument("--keep-scratch", action="store_true", help="keep the scratch directory after the run")
    p.add_argument("--keep-calibration", action="store_true",
                   help="keep the finished calibration after a rebuild of the binaries that does not change the numbers")
    p.add_argument("--no-space-check", action="store_true")
    # a large MoE needs more than 10 minutes (GLM 5.3 on 3 CMP: about 13); the cap only stops a hung measurement
    p.add_argument("--measurement-seconds", type=float, default=3600, help="cap of the operation measurement in prepare")
    p.add_argument("--converter-python", help="python that runs convert_hf_to_gguf.py (default: this one)")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--verify-timeout", type=float, default=3600)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        {"capacity": command_capacity, "setup": command_setup, "plan": command_plan, "show": command_show, "encode": command_encode, "run": command_run}[args.command](args)
    except (QuantError, ValueError, OSError, KeyError) as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
