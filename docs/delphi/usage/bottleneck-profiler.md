# Bottleneck profiler recipes

`tools/bottlenecks/profile.py` runs a recipe-defined DELPHI benchmark, collects an execution trace, and writes normalized JSON plus raw artifacts. It uses the existing execution-trace profiler. It does not change scheduling, build code, or backend behavior.

The public runner has no model paths or deployment policy. Keep local model paths, hardware policy, and production recipes in a separate local Python file.

## Minimal recipe

```python
from tools.bottlenecks.profile import BenchArm, Recipe


def example() -> Recipe:
    return Recipe(
        name="example",
        description="One local model on one GPU.",
        arms=(
            BenchArm(
                name="plain",
                command=("build/bin/llama-bench", "-m", "/models/model.gguf", "-p", "512", "-n", "32", "-ngl", "999", "-r", "5"),
                model_paths=("/models/model.gguf",),
            ),
        ),
    )
```

Run it from the repository root:

```sh
python3 tools/bottlenecks/profile.py \
  --recipe /path/to/recipes.py:example \
  --bindir build/bin \
  --output /tmp/example.json \
  --print-summary
```

`llama-bench` repetitions remain in one process. This loads the model once per arm. The runner adds JSON output if the command does not set it.

`--bindir DIR` replaces the direct `llama-bench` or `llama-server` executable in every selected recipe arm with `DIR/<binary-name>`. It does not alter helper commands such as Python or SPEED-Bench. Use it to measure an agent's own build directory, for example `--bindir build-moe-experiment/bin`; use `--bindir build/bin` for the canonical build.

## One-off comparisons

Use `--set ARM.OPTION=VALUE` to replace an option in a benchmark or server command. `all` changes every arm. Use `--env NAME=VALUE` for an explicit runtime environment override.

```sh
python3 tools/bottlenecks/profile.py \
  --recipe /path/to/recipes.py:example \
  --set plain.-ub=2048 \
  --env GGML_CUDA_MOE_GROUPED=0 \
  --output /tmp/example-ub2048.json \
  --print-summary
```

The JSON records the complete command, selected environment, binary version, model file size and modification time, Git identity, raw benchmark output, and trace-derived data. Point the recipe at an alternate binary when a comparison needs a different build. The profiler does not build binaries itself.

## Server and speculative recipes

Use `ServerArm` for one `llama-server` load and one SPEED-Bench client invocation. Its `server_command` and `client_command` can contain `{port}`, `{url}`, `{output}`, and `{artifact_dir}`. The runner waits for `/health`, runs the optional warmup command, invokes the client, and stops the server with `SIGINT` so the trace is finalized.

Use the same server workload for a plain arm and a speculative arm. For execution-cost comparisons, a fixed synthetic acceptance pattern removes prompt-dependent acceptance variance. Also collect a short normal-acceptance arm when a change can affect draft quality, logits, sampling, or acceptance.

## Interpretation

The terminal summary is intentionally terse and suitable for an engineering prompt. It is not a license to add all displayed rows. Host scheduler spans can overlap device work; device submit spans can overlap across backends; and operation spans from graph-disabled runs are diagnostic. Use the wall-time row for an optimistic Amdahl calculation and state the evidence class used.

Use `tools/exec-trace/exec-trace-integrity.py` when a `llama-bench` trace must be established as production-representative. Its seven-process GPU-timing sequence intentionally reloads the model, so run it on build, profiler, or deployment-configuration changes instead of treating it as a cheap repetition mechanism.
