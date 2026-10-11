#!/usr/bin/env python3
"""Offline V1 quantization advisor: inspect, prepare, plan, encode and validate."""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from config import AXES, SCHEMA, QuantError, digest, file_identity, load_config, write_json
from model import classify, merge_imatrices, read_model
from quality import apply_quality, load_quality, quality_metadata, repair_options, verify_quality_output
from search import make_options, memory, search
from timing import MODEL_REVISION, Predictor, fit_slopes, gpu, machine_settings, measure_ops, profile_model, profile_times, run


def inputs(cfg, info):
    result = {"model": info["identity"], "imatrix": [file_identity(path, content=True) for path in cfg["imatrix"]],
            "sampling": cfg["sampling"], "groups": info["groups"],
            "binaries": {name: file_identity(Path(cfg["bindir"]) / name) for name in
                         (("llama-quantize", "test-backend-ops", "llama-bench") if "quality" in cfg else
                          ("llama-quant-damage", "llama-quantize", "test-backend-ops", "llama-bench"))},
            "libraries": [file_identity(p) for p in sorted(Path(cfg["bindir"]).glob("lib*.so*")) if not p.is_symlink()],
            "runtime": cfg["runtime"], "timing": cfg["timing"], "timing_model_revision": MODEL_REVISION,
            "gpu": gpu(cfg), "ram_routed": "RAM" in cfg["devices"], "cache_mib": cfg["cache_mib"],
            "machine_settings": machine_settings(),
            "inherited_environment": {name: value for name, value in os.environ.items()
                                      if name.startswith(("GGML_", "LLAMA_", "CUDA_", "HIP_", "HSA_", "ROCR_", "OMP_", "LD_LIBRARY_PATH"))}}
    if "quality" in cfg:
        result["quality"] = file_identity(cfg["quality"]["table"], content=True)
    return result


def configured_model(cfg):
    info = classify(read_model(cfg["model"]), cfg)
    table = load_quality(info, cfg)
    return apply_quality(info, table), table


def get_damage(info, cfg, directory, imatrix):
    cache_key = digest({"model": info["identity"], "sampling": cfg["sampling"], "groups": info["groups"],
                        "imatrices": [file_identity(p, content=True) for p in cfg["imatrix"]],
                        "binary": file_identity(Path(cfg["bindir"]) / "llama-quant-damage"),
                        "libraries": [file_identity(p) for p in sorted(Path(cfg["bindir"]).glob("lib*.so*")) if not p.is_symlink()],
                        "threads": cfg["runtime"]["threads"]})
    cache_file = directory / "damage-cache.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text())
        if cached["key"] == cache_key:
            return [json.loads(Path(path).read_text()) for path in cached["artifacts"]]
    artifacts = []
    paths = []
    for category in ("primary", "other"):
        selected = [t for t in info["tensors"] if t["category"] == category and not t["preserved"]
                    and any(q != t["source_dtype"] for q in t["pool"])]
        if not selected:
            continue
        qtypes = sorted({q for t in selected for q in t["pool"]})
        command = [str(Path(cfg["bindir"]) / "llama-quant-damage"), "-m", cfg["model"], "--qtypes", ",".join(qtypes),
                   "--tensor", "^(" + "|".join(re.escape(t["name"]) for t in selected) + ")$", "--min-elements", "1",
                   "--samples", str(cfg["sampling"]["samples"]), "--max-samples", str(cfg["sampling"]["max_samples"]),
                   "--seed", str(cfg["sampling"]["seed"]), "-t", str(cfg["runtime"]["threads"]),
                   "--json", str(directory / f"damage-{category}.json")]
        if imatrix:
            command += ["--imatrix", imatrix]
        if not cfg["sampling"]["row_stats"]:
            command += ["--no-row-stats"]
        print(f"Sampling {category}: {len(selected)} tensors, {len(qtypes)} qtypes", flush=True)
        run(command, directory / f"damage-{category}.log", cfg["runtime"]["env"])
        artifact = json.loads((directory / f"damage-{category}.json").read_text())
        artifacts.append(artifact)
        paths.append(str(directory / f"damage-{category}.json"))
    write_json(cache_file, {"key": cache_key, "artifacts": paths})
    return artifacts


def tensor_plan(info, choices):
    return {name: choices[group["name"]] for group in info["groups"] for name in group["members"]}


def verify_output(info, types, output, check_preserved=True):
    actual = read_model(output)
    expected = {t["name"]: t for t in info["tensors"]}
    found = {t["name"]: t for t in actual["tensors"]}
    if set(expected) != set(found):
        raise QuantError("encoded model tensor names differ from the source")
    for name, t in found.items():
        if t["source_dtype"] != types[name] or t["shape"] + [1] * (4 - len(t["shape"])) != expected[name]["shape"] + [1] * (4 - len(expected[name]["shape"])):
            raise QuantError(f"quantizer changed the planned shape/type of {name}: {t['source_dtype']} versus {types[name]}")
        source = expected[name]
        if check_preserved and source["preserved"]:
            if source["source_bytes"] != t["source_bytes"]:
                raise QuantError(f"preserved tensor size changed: {name}")
            with open(source["file"], "rb") as before, open(t["file"], "rb") as after:
                before.seek(source["offset"])
                after.seek(t["offset"])
                left = t["source_bytes"]
                while left:
                    n = min(left, 1 << 20)
                    if before.read(n) != after.read(n):
                        raise QuantError(f"preserved tensor values changed: {name}")
                    left -= n
    return actual


def encode(info, cfg, choices, imatrix, output, quality=None):
    output = Path(output).resolve()
    if output.exists():
        raise QuantError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    types = tensor_plan(info, choices)
    plan_file = output.with_suffix(".tensor-types.txt")
    plan_file.write_text("".join(f"^{re.escape(name)}$={dtype.lower()}\n" for name, dtype in types.items()))
    partial = output.with_suffix(".partial.gguf")
    if partial.exists():
        raise QuantError(f"partial output exists; inspect it before retrying: {partial}")
    command = [str(Path(cfg["bindir"]) / "llama-quantize"), "--pure", "--tensor-type-file", str(plan_file)]
    if imatrix:
        command += ["--imatrix", str(imatrix)]
    if cfg.get("allow_requantize", False):
        command += ["--allow-requantize"]
    command += [cfg["model"], str(partial), "Q2_K", str(cfg["runtime"]["threads"])]
    print(f"Encoding {output.name}", flush=True)
    run(command, output.with_suffix(".encode.log"), cfg["runtime"]["env"])
    verify_output(info, types, partial)
    verify_quality_output(info, quality, types, partial)
    partial.replace(output)
    write_json(output.with_suffix(".encode.json"), {"command": command, "types": types, "source": info["identity"],
                                                  "output": file_identity(output), "verified_preserved_payloads": True,
                                                  "quality": quality_metadata(quality)})
    return str(output)


def prepare(cfg, directory, timeout):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    listed = subprocess.check_output([str(Path(cfg["bindir"]) / "llama-bench"), "--list-devices"], stderr=subprocess.STDOUT,
                                     env={**os.environ, **cfg["runtime"]["env"]}).decode()
    for device in cfg["devices"]:
        if device != "RAM" and not re.search(r"\b" + re.escape(device) + r"\b", listed):
            raise QuantError(f"unavailable device {device}")
    info, quality = configured_model(cfg)
    if any("engram_embd" in t["name"] or "per_layer_token_embd" in t["name"] for t in info["tensors"]):
        raise QuantError("lazy-table residency planning needs explicit resident-byte accounting and is not implemented")
    if len([name for name in cfg["devices"] if name != "RAM"]) > 1 and "RAM" not in cfg["devices"]:
        raise QuantError("GPU-only multiple-device placement is not implemented; use one GPU or a RAM-routed speed projection")
    key = digest(inputs(cfg, info))
    snapshot = directory / "prepared.json"
    if snapshot.exists():
        old = load_prepared(snapshot, cfg)
        print(f"Reusing {snapshot}; timing reflects the recorded machine state", flush=True)
        return old
    imatrix = merge_imatrices(cfg["imatrix"], directory / "merged-imatrix.gguf")
    damage = [] if quality is not None else get_damage(info, cfg, directory, imatrix)
    print("Measuring bounded operation signatures", flush=True)
    operations = measure_ops(info, cfg, directory / "calibration", timeout)
    excluded = repair_options(info, quality, operations, cfg) if quality is not None else make_options(info, damage, operations, cfg)
    baseline_choices = {group["name"]: min(group["options"], key=lambda o: (o["bytes"], o["damage"], o["qtype"]))["qtype"]
                        for group in info["groups"]}
    mem = memory(info, cfg, baseline_choices)
    if any(value < 0 for value in mem["headroom_bytes"].values()):
        raise QuantError("minimum-byte baseline exceeds a device budget")
    baseline = directory / "baseline.gguf"
    if baseline.exists():
        verify_output(info, tensor_plan(info, baseline_choices), baseline)
        verify_quality_output(info, quality, tensor_plan(info, baseline_choices), baseline)
    elif quality is not None:
        baseline = Path(quality["floor"]["path"])
        identity = file_identity(baseline, content=True)
        if identity["sha256"] != quality["floor"]["sha256"]:
            raise QuantError("repair floor content changed")
        verify_output(info, tensor_plan(info, baseline_choices), baseline)
        verify_quality_output(info, quality, tensor_plan(info, baseline_choices), baseline)
    else:
        encode(info, cfg, baseline_choices, imatrix, baseline)
    print("Profiling baseline short/long prefill and decode", flush=True)
    profile = profile_model(cfg, baseline, directory / "baseline-profile.json")
    result = {"schema": SCHEMA, "input_key": key, "config": cfg, "model": info, "damage": damage,
              "operations": operations, "excluded": excluded, "imatrix": imatrix,
              "baseline": {"path": str(baseline), "choices": baseline_choices, "measured_ms": profile_times(profile, cfg),
                           "memory": mem, "profile": str(directory / "baseline-profile.json")},
              "slopes": {regime: 1.0 for regime in AXES[1:]}, "timing_qualified": False}
    if quality is not None:
        result["quality"] = quality
    write_json(snapshot, result)
    return result


def load_prepared(path, cfg):
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise QuantError("unsupported prepared schema")
    info, quality = configured_model(cfg)
    facts = inputs(cfg, info)
    if data["input_key"] != digest(facts):
        # Early V1 snapshots omitted the cost-model settings from their key. Upgrade only equivalent inputs.
        legacy = dict(facts)
        legacy.pop("timing")
        if data["input_key"] == digest(legacy) and cfg["timing"] == data["config"].get("timing", {}):
            data["input_key"] = digest(facts)
        else:
            raise QuantError("model, imatrix, candidates, binary, cost model or runtime changed; prepare again")
    return data


def predictor_for(data, cfg):
    baseline = data["baseline"]
    if "cpu_compute_ms" not in baseline:
        profile_file = Path(baseline["profile"])
        profile = json.loads(profile_file.read_text()) if profile_file.exists() else {}
        baseline["cpu_compute_ms"] = (profile.get("arms", [{}])[0].get("trace", {}).get("analysis", {})
                                      .get("decode", {}).get("cpu_compute_ms", {}).get("median"))
    predictor = Predictor(data["model"], cfg, data["operations"], data["baseline"]["choices"],
                          data["baseline"]["measured_ms"], data["slopes"], baseline["cpu_compute_ms"])
    predictor.baseline_cost = predictor.cost(data["baseline"]["choices"], data["baseline"]["memory"]["cache_hit_prior"])
    return predictor


def plan(data, cfg, output):
    result = search(data["model"], cfg, predictor_for(data, cfg), data["baseline"]["choices"])
    outdir = Path(output).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    recipes = []
    for index, state in enumerate(result.pop("frontier")):
        state.pop("key")
        state["id"] = f"recipe-{index:03d}"
        state["tensor_types"] = tensor_plan(data["model"], state["choices"])
        write_json(outdir / f"{state['id']}.json", state)
        recipes.append(state)
    projection = sum(name != "RAM" for name in cfg["devices"]) > 1 and any(cfg["multi_gpu_speedup_percent"].values())
    qualified = data["timing_qualified"] and not projection
    payload = {"schema": SCHEMA, "config": cfg, "source": data["model"]["identity"], "prepared": data,
               "search": result, "recipes": recipes, "timing_qualified": qualified,
               "single_gpu_timing_qualified": data["timing_qualified"], "user_gpu_projection": projection,
               "quality": quality_metadata(data.get("quality")),
               "limitations": ["repair loss is an experimental conditional KL sum, not a bound on original KLD or accuracy" if "quality" in cfg else "damage is a local source-relative proxy, not KLD or accuracy",
                               "uniform synthetic routing and fixed route frequencies",
                               "link bandwidth and handoff latency are priors unless supplied",
                               "cache hits use a uniform residency prior; cache policy replay is not calibrated",
                               "tail ubatches use interpolation; trace integrity has not been certified",
                               "multiple GPUs use user throughput factors, without pooled capacity or a runtime layout"]}
    payload["effective_cpu_decode_gb_s"] = (predictor_for(data, cfg).cpu_bytes_per_ms or 0) / 1e6
    write_json(outdir / "frontier.json", payload)
    print(f"{result['status']}: {len(recipes)} nondominated evaluated recipes; {result['evaluated']} evaluations")
    print(f"recipe       {'repair loss' if 'quality' in cfg else 'damage':12s}  decode t/s  short t/s  long t/s  score")
    for state in recipes[:cfg["search"]["display"]]:
        times = state["predicted_ms"]
        print(f"{state['id']}  {state['damage']:12.5g}  {1000/times['decode']:10.2f}  "
              f"{1000*cfg['runtime']['short_tokens']/times['short']:9.2f}  "
              f"{1000*cfg['runtime']['long_tokens']/times['long']:8.2f}  {state['score']:.4f}")
    status = "user GPU projection from the single-GPU baseline" if projection else "qualified on held-out witnesses" if qualified else "unqualified estimates"
    print(f"Saved {outdir / 'frontier.json'}; predictions are {status}")
    if "quality" in cfg:
        print("Quality uses whole-group conditional repair KL; max_loss is a proxy ceiling, not an original-KLD bound")
    return payload


def witness_choices(info, baseline, spec):
    choices = dict(baseline)
    by_name = {t["name"]: t for t in info["tensors"]}
    for group in info["groups"]:
        # Category witnesses change calibrated groups only. Other floor tensors must stay fixed.
        if "execution_groups" in info and "repair_options" not in group:
            continue
        selected = spec.get("primary" if any(by_name[n]["category"] == "primary" for n in group["members"]) else "other")
        if selected and all(not by_name[n]["preserved"] for n in group["members"]):
            selected = selected.upper()
            if selected not in group["pool"]:
                raise QuantError(f"witness selects unmeasured type {selected}: {group['name']}")
            choices[group["name"]] = selected
    for pattern, dtype in spec.get("overrides", {}).items():
        for group in info["groups"]:
            if re.search(pattern, group["name"]):
                if dtype.upper() not in group["pool"]:
                    raise QuantError(f"witness override is outside group pool: {group['name']}")
                choices[group["name"]] = dtype.upper()
    return choices


def validate(data, cfg, specification, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    specs = json.loads(Path(specification).read_text())
    witnesses, seen_mixes = [], set()
    for spec in specs:
        if spec.get("role") not in ("fit", "holdout"):
            raise QuantError("witness role must be fit or holdout")
        name = spec["name"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise QuantError("validation names must be filename-safe")
        choices = witness_choices(data["model"], data["baseline"]["choices"], spec)
        mem = memory(data["model"], cfg, choices)
        mix_key = digest(choices)
        if mix_key in seen_mixes:
            raise QuantError("validation mixes must be distinct, including across fit and holdout roles")
        seen_mixes.add(mix_key)
        if any(value < 0 for value in mem["headroom_bytes"].values()):
            raise QuantError(f"witness {name} exceeds budget")
        model = directory / f"{name}.gguf"
        if model.exists():
            verify_output(data["model"], tensor_plan(data["model"], choices), model)
            verify_quality_output(data["model"], data.get("quality"), tensor_plan(data["model"], choices), model)
        else:
            encode(data["model"], cfg, choices, data["imatrix"], model, data.get("quality"))
        profile_file = directory / f"{name}-profile.json"
        if profile_file.exists():
            profile = json.loads(profile_file.read_text())
        else:
            print(f"Profiling witness {name} ({spec['role']})", flush=True)
            profile = profile_model(cfg, model, profile_file)
        witnesses.append({"name": name, "role": spec["role"], "choices": choices, "hit_fraction": mem["cache_hit_prior"],
                          "measured": profile_times(profile, cfg), "profile": str(profile_file)})
    # Witnesses execute on the leading GPU. Extra-GPU factors remain assumptions, not measurements.
    single_cfg = copy.deepcopy(cfg)
    single_cfg["multi_gpu_speedup_percent"] = {regime: 0 for regime in AXES[1:]}
    predictor = predictor_for(data, single_cfg)
    training = [w for w in witnesses if w["role"] == "fit"]
    holdout = [w for w in witnesses if w["role"] == "holdout"]
    if len(training) < 2 or len(holdout) < 2:
        raise QuantError("validation needs at least two fit and two holdout mixes")
    try:
        data["slopes"] = fit_slopes(predictor, training)
        fit_error = None
    except QuantError as exc:
        data["slopes"] = {regime: 1.0 for regime in AXES[1:]}
        fit_error = str(exc)
    predictor = predictor_for(data, single_cfg)
    for w in witnesses:
        try:
            w["predicted"] = predictor.predict(w["choices"], w["hit_fraction"])
            w["relative_error"] = {regime: abs(w["predicted"][regime] / w["measured"][regime] - 1) for regime in AXES[1:]}
        except QuantError as exc:
            w["prediction_error"] = str(exc)
            w["relative_error"] = {regime: 1.0 for regime in AXES[1:]}
    rank_errors = []
    for regime in AXES[1:]:
        for i, a in enumerate(holdout):
            for b in holdout[i + 1:]:
                if "predicted" not in a or "predicted" not in b:
                    continue
                difference = a["measured"][regime] / b["measured"][regime] - 1
                if abs(difference) >= 0.05 and difference * (a["predicted"][regime] - b["predicted"][regime]) < 0:
                    rank_errors.append({"regime": regime, "a": a["name"], "b": b["name"]})
    data["timing_qualified"] = (fit_error is None and not rank_errors and
                                all(max(w["relative_error"].values()) <= 0.15 for w in holdout))
    report = {"schema": SCHEMA, "slopes": data["slopes"], "fit_error": fit_error, "rank_errors": rank_errors,
              "timing_qualified": data["timing_qualified"], "witnesses": witnesses}
    write_json(directory / "validation.json", report)
    print(f"Held-out timing gate: {'passed' if data['timing_qualified'] else 'not passed'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "prepare", "plan", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        if name in ("prepare", "plan", "validate"):
            p.add_argument("--output", type=Path, required=True)
        if name in ("plan", "validate"):
            p.add_argument("--prepared", type=Path, required=True)
        if name == "prepare":
            p.add_argument("--measurement-seconds", type=float, default=600)
        if name == "validate":
            p.add_argument("--mixes", type=Path, required=True)
    p = sub.add_parser("encode")
    p.add_argument("--frontier", type=Path, required=True)
    p.add_argument("--recipe", default="recipe-000")
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "encode":
            frontier = json.loads(args.frontier.read_text())
            if frontier.get("schema") != SCHEMA:
                raise QuantError("unsupported frontier schema")
            cfg = frontier["config"]
            info = frontier["prepared"]["model"]
            if read_model(cfg["model"])["identity"] != info["identity"]:
                raise QuantError("source model changed since planning")
            current, quality = configured_model(cfg)
            if digest(inputs(cfg, current)) != frontier["prepared"]["input_key"]:
                raise QuantError("encoder, imatrix or runtime inputs changed since planning")
            recipe = next((r for r in frontier["recipes"] if r["id"] == args.recipe), None)
            if recipe is None:
                raise QuantError(f"unknown recipe {args.recipe}")
            encode(info, cfg, recipe["choices"], frontier["prepared"]["imatrix"], args.output, quality)
            return
        cfg = load_config(args.config)
        if args.command == "inspect":
            info, quality = configured_model(cfg)
            print(json.dumps({"model": info["architecture"], "devices_bytes": cfg["devices"],
                              "classification": {c: sum(t["source_bytes"] for t in info["tensors"] if t["category"] == c)
                                                 for c in ("primary", "other")}, "groups": info["groups"]}, indent=2))
        elif args.command == "prepare":
            prepare(cfg, args.output, args.measurement_seconds)
        else:
            data = load_prepared(args.prepared, cfg)
            if args.command == "plan":
                plan(data, cfg, args.output)
            else:
                validate(data, cfg, args.mixes, args.output)
                write_json(args.prepared, data)
    except (ValueError, OSError, KeyError, OverflowError, subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
