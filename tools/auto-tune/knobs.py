"""Registry of the launch knobs that the auto-tuner may change.

A knob is a server option, a set of server options, a presence switch or an environment variable. The registry holds only
knobs that are worth tuning; debug, trace and A/B off-switches are not in it. Classes:

- fit: placement and memory (batch sizes, staging slots, cache margin). A level that does not load is a measured boundary.
- speed: speed only; the output stays within the base numerics.
- accept: speculative decoding settings; the output is the same, the acceptance changes.
- precision: opt-in precision trades. A config with a non-exact level needs a KLD measurement under the limit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

CLASSES = ("fit", "speed", "accept", "precision")
OFF = "off"
ON = "on"


@dataclass(frozen=True)
class Context:
    """Facts that levels and conditions depend on."""
    command: tuple[str, ...]
    env: Mapping[str, str]
    physical_cores: int
    max_threads: int

    def option(self, name: str) -> str | None:
        return option_value(self.command, name)

    def has(self, name: str) -> bool:
        return name in self.command


@dataclass(frozen=True)
class Knob:
    name: str
    cls: str
    kind: str  # option, options, switch, env
    targets: tuple[str, ...]  # options or the env variable
    levels: Sequence[str] | Callable[[Context], Sequence[str]]
    doc: str
    requires: Callable[[Context], bool] = lambda ctx: True
    exact: str | None = None  # precision knobs: the level without precision loss
    default: bool = True  # False: only tuned when the config includes it (opt-in)
    unset: str | None = None  # the value of an env knob or an option that is not set (its built-in default)

    def applies(self, ctx: Context) -> bool:
        return self.requires(ctx)

    def level_list(self, ctx: Context) -> list[str]:
        values = self.levels(ctx) if callable(self.levels) else self.levels
        return [str(v) for v in values]

    def current(self, command: Sequence[str], env: Mapping[str, str]) -> str | None:
        if self.kind == "env":
            return env.get(self.targets[0], self.unset)
        if self.kind == "switch":
            return ON if self.targets[0] in command else OFF
        value = option_value(command, self.targets[0])
        return self.unset if value is None else value

    def apply(self, command: Sequence[str], env: Mapping[str, str], value: str | None) -> tuple[tuple[str, ...], dict[str, str]]:
        """Return the command and env with this knob at *value*; None removes it (built-in default)."""
        cmd = tuple(command)
        out_env = dict(env)
        if self.kind == "env":
            if value is None:
                out_env.pop(self.targets[0], None)
            else:
                out_env[self.targets[0]] = value
        elif self.kind == "switch":
            cmd = remove_option(cmd, self.targets[0], has_value=False)
            if value == ON:
                cmd = (*cmd, self.targets[0])
        else:
            for target in self.targets:
                cmd = set_option(cmd, target, value)
        if self.name == "-ub" and value is not None:
            batch = option_value(cmd, "-b")
            if batch is not None and int(batch) < int(value):
                cmd = set_option(cmd, "-b", value)
        return cmd, out_env


def option_value(command: Sequence[str], option: str) -> str | None:
    for index, item in enumerate(command):
        if item == option and index + 1 < len(command):
            return command[index + 1]
        if option.startswith("--") and item.startswith(option + "="):
            return item.split("=", 1)[1]
    return None


def remove_option(command: Sequence[str], option: str, *, has_value: bool = True) -> tuple[str, ...]:
    out: list[str] = []
    skip = False
    for item in command:
        if skip:
            skip = False
            continue
        if item == option:
            skip = has_value
            continue
        if has_value and option.startswith("--") and item.startswith(option + "="):
            continue
        out.append(item)
    return tuple(out)


def set_option(command: Sequence[str], option: str, value: str | None) -> tuple[str, ...]:
    """Replace the value of *option*, append the option if it is missing, or remove it when *value* is None."""
    if value is None:
        return remove_option(command, option)
    out = list(command)
    for index, item in enumerate(out):
        if item == option and index + 1 < len(out):
            out[index + 1] = value
            return tuple(out)
        if option.startswith("--") and item.startswith(option + "="):
            out[index] = f"{option}={value}"
            return tuple(out)
    return (*out, option, value)


def _speculative(ctx: Context) -> bool:
    return ctx.has("--spec-type") and ctx.option("--spec-synth-rates") is None


def _adaptive(ctx: Context) -> bool:
    return _speculative(ctx) and ctx.has("--spec-draft-adaptive")


def _host_experts(ctx: Context) -> bool:
    return ctx.has("-cmoe") or ctx.has("--cpu-moe") or ctx.has("-ncmoe") or ctx.has("--n-cpu-moe")


def _staging(ctx: Context) -> bool:
    return ctx.env.get("GGML_SCHED_STAGE") == "1"


def _cache(ctx: Context) -> bool:
    return _host_experts(ctx) and ctx.env.get("LLAMA_MOE_CACHE_MIB", "auto") not in ("0",)


def _multi_device(ctx: Context) -> bool:
    devices = ctx.option("--device") or ctx.option("-dev") or ""
    return len([d for d in devices.split(",") if d]) > 1


def _threads(ctx: Context) -> list[int]:
    # GPU driver threads need free cores: never all of them (64 threads on the 64-core host stalls the GPUs)
    levels = [16, 24, 32, 40, 48, 56, 64, 96, 128]
    return [t for t in levels if t <= ctx.max_threads]


REGISTRY: tuple[Knob, ...] = (
    # fit
    Knob("-ub", "fit", "option", ("-ub",), (256, 512, 1024, 2048, 4096), "physical batch; -b is raised to it if lower"),
    Knob("-b", "fit", "option", ("-b",), (2048, 4096, 8192, 10240), "logical batch", requires=lambda c: c.option("-ub") is not None),
    Knob("-ubd", "fit", "option", ("-ubd",), (128, 512, 2048), "draft context physical batch (MTP/DSpark prompt ingestion)", requires=_speculative),
    Knob("GGML_SCHED_STAGE_SLOTS", "fit", "env", ("GGML_SCHED_STAGE_SLOTS",), (2, 4, 5, 6), "staging ring slots per device", requires=_staging, unset="4"),
    Knob("LLAMA_MOE_CACHE_AUTO_MARGIN_MIB", "fit", "env", ("LLAMA_MOE_CACHE_AUTO_MARGIN_MIB",), (768, 1536, 3072),
         "VRAM left free by the auto expert cache", requires=_cache, unset="1536"),
    # speed
    Knob("-t", "speed", "option", ("-t",), _threads, "CPU threads"),
    Knob("GGML_SCHED_STAGE_MIN_TOKENS", "speed", "env", ("GGML_SCHED_STAGE_MIN_TOKENS",), (512, 1024, 2048),
         "smallest batch whose host experts stream to a GPU", requires=_staging, unset="512"),
    Knob("GGML_SCHED_STAGE_REGISTER", "speed", "env", ("GGML_SCHED_STAGE_REGISTER",), (0, 1), "page-lock staged host weights", requires=_staging, unset="0"),
    Knob("LLAMA_MOE_CACHE_PROMPT_ROWS", "speed", "env", ("LLAMA_MOE_CACHE_PROMPT_ROWS",), (0, 255, 1023),
         "prompt batches up to this many rows also use the GPU expert cache", requires=_cache, unset="0"),
    Knob("LLAMA_MOE_CACHE_EVERY", "speed", "env", ("LLAMA_MOE_CACHE_EVERY",), (4, 8, 16), "small batches between cache refreshes", requires=_cache, unset="8"),
    # accept
    Knob("--spec-draft-n-max", "accept", "option", ("--spec-draft-n-max",), (2, 3, 4, 5, 6), "largest draft", requires=_speculative),
    Knob("--spec-draft-adaptive", "accept", "switch", ("--spec-draft-adaptive",), (OFF, ON), "adaptive draft depth", requires=_speculative),
    Knob("--spec-draft-n-min-adaptive", "accept", "option", ("--spec-draft-n-min-adaptive",), (1, 2), "smallest adaptive draft", requires=_adaptive),
    # precision
    Knob("kv-type", "precision", "options", ("-ctk", "-ctv"), ("f16", "q8_0"), "KV cache type (needs -fa on)", exact="f16", unset="f16",
         requires=lambda c: c.option("-fa") in ("on", "1")),
    Knob("LLAMA_MOE_CACHE_MIB", "precision", "env", ("LLAMA_MOE_CACHE_MIB",), (0, "auto"),
         "GPU expert cache (its GPU part quantizes the activations: KLD about 0.013 on DeepSeek)", requires=_host_experts, exact="0", unset="auto"),
    Knob("-sm", "precision", "option", ("-sm",), ("layer", "tensor"), "split mode (tensor: more speed, heat and power; changes numerics)",
         requires=_multi_device, exact="layer", unset="layer", default=False),
    Knob("LLAMA_ATTN_SPLIT_INDEXER", "precision", "env", ("LLAMA_ATTN_SPLIT_INDEXER",), (0, 1), "second GPU scores half the indexer rows",
         requires=lambda c: "LLAMA_ATTN_SPLIT" in c.env, exact="0", unset="0", default=False),
)

BY_NAME = {knob.name: knob for knob in REGISTRY}


def select(ctx: Context, *, only: Sequence[str] = (), include: Sequence[str] = (), exclude: Sequence[str] = (),
           levels: Mapping[str, Sequence] | None = None) -> list[tuple[Knob, list[str]]]:
    """The knobs to tune for this command, each with its levels (the current level first). *only* limits the set, *include*
    adds opt-in knobs, *exclude* removes knobs; *levels* replaces the registry levels of a knob."""
    unknown = sorted((set(only) | set(include) | set(exclude) | set(levels or {})) - set(BY_NAME))
    if unknown:
        raise ValueError(f"unknown knob(s): {', '.join(unknown)}; known: {', '.join(BY_NAME)}")
    chosen = []
    for knob in REGISTRY:
        if knob.name in exclude or (only and knob.name not in only):
            continue
        if not knob.default and knob.name not in include and knob.name not in only:
            continue
        if not knob.applies(ctx):
            continue
        values = [str(v) for v in (levels or {}).get(knob.name, ())] or knob.level_list(ctx)
        current = knob.current(ctx.command, ctx.env)
        if current is not None and current not in values:
            values.append(current)
        if current is not None:
            values.remove(current)
            values.insert(0, current)
        if len(values) > 1:
            chosen.append((knob, values))
    order = {name: i for i, name in enumerate(CLASSES)}
    chosen.sort(key=lambda item: order[item[0].cls])
    return chosen


def exact_settings(ctx: Context) -> list[tuple[Knob, str]]:
    """Precision knobs that apply to this command, at their exact level."""
    return [(knob, knob.exact) for knob in REGISTRY if knob.cls == "precision" and knob.exact is not None and knob.applies(ctx)]


def is_exact(ctx: Context) -> bool:
    return all(knob.current(ctx.command, ctx.env) == exact for knob, exact in exact_settings(ctx))


def precision_key(ctx: Context) -> dict[str, str | None]:
    return {knob.name: knob.current(ctx.command, ctx.env) for knob, _ in exact_settings(ctx)}
