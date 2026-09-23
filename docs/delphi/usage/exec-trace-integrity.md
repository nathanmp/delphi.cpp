# exec-trace-integrity: validate that a trace is production-representative

`tools/exec-trace/exec-trace-integrity.py` compares a clean `llama-bench` run with an equivalent execution-traced run and reports whether the trace is safe to use for production timing.

## Synopsis

```text
python3 tools/exec-trace/exec-trace-integrity.py \
    --binary build/bin/llama-bench \
    --report /tmp/integrity.json \
    --launcher "numactl --cpunodebind=0 --membind=0" \
    -- -m model.gguf -p 512 -n 0
```

Arguments after `--` are passed to `llama-bench`. The harness forces JSON benchmark output and exits with status 2 when the safe trace or the requested timing mode is not trustworthy.

## Options

| flag | meaning |
| --- | --- |
| `--binary PATH` | `llama-bench` executable (required) |
| `--report PATH` | machine-readable JSON report (required); the raw trace defaults to beside the report |
| `--trace PATH` | raw trace path |
| `--repetitions N` | `llama-bench` repetitions per process (default 5) |
| `--max-events N` | set `GGML_EXEC_TRACE_MAX_EVENTS` for the trace runs |
| `--launcher CMD` | command prefix, such as a profiler, parsed with shell quoting |
| `--gpu-timing {submit,op}` | also run a safe trace with `GGML_EXEC_TRACE_GPU_TIMING` set to this mode |
| `--gpu-timing-events N` | set `GGML_EXEC_TRACE_GPU_TIMING_EVENTS` for the timing runs |

## Run sequence

The harness uses the same binary and arguments for five processes (seven with `--gpu-timing`):

1. A trace-disabled structural witness with `GGML_EXEC_TRACE_INTEGRITY` enabled. This run also warms file-backed routed-expert pages before the timed pair.
2. A clean timed baseline with tracing and the integrity observer disabled.
3. A timed trace run with the structural observer enabled.
4. A second timed trace run.
5. A second clean timed baseline.

With `--gpu-timing submit` or `--gpu-timing op`, two timing runs (`GGML_EXEC_TRACE` plus `GGML_EXEC_TRACE_GPU_TIMING`) run between the two trace runs, so the timed order is clean, trace, timing, timing, trace, clean. `--gpu-timing-events` sets the event pool size for these runs.

The separate structural witness prevents observer work from being hidden in the measured trace overhead. Reversing the second timed pair balances process-order effects such as page-cache, thermal, and GPU-clock drift. The machine-readable JSON report contains the exact commands, relevant environment, process wall times, every `llama-bench` wall-time and throughput sample, median/min/max values, and sample dispersion. The human-readable output shows the main gate result.

## Gate

The gate compares ordered graph shape/operation data, total node count, scheduler split structure, node-to-backend assignments, scheduler copy paths, backend dispatch choices, GPU graph lifecycle behavior, and scheduler evaluation-callback use. A structural mismatch rejects the trace even when throughput is unchanged. A GPU graph replay-to-launch fallback is therefore a failure, not harmless timing noise.

Throughput classification uses the median sample on each side:

- At most 1% slowdown: preferred and production-safe.
- More than 1% and at most 2%: acceptable for Milestone 1.
- More than 2%: investigate; the report is not production-representative.

With `--gpu-timing`, the report has a `gpu_timing` section that compares the timing runs with the clean runs and with the safe trace, checks every structural fingerprint against the witness, and reports how many spans resolved, were not recorded, stayed unresolved, or failed, and whether each epoch has a consistent host placement. The classification is `low-perturbation` (at most 2% slowdown, complete coverage), `low-perturbation, incomplete coverage`, `perturbative`, or `invalid`. The `GGML_EXEC_TRACE_GPU_TIMING` runtime flag is expected in the timing runs and is not treated as contamination there.

The report also rejects truncated traces, dropped events, debug or sanitizer builds detected by the runtime or adjacent CMake cache, graph-disable and operation-trace environment settings, directly linked profiler libraries, and an external profiler named in `--launcher`. It ignores profiler support libraries pulled in transitively by a normal GPU runtime. Detection is not exhaustive. Driver-level tracing, system profilers attached after launch, thermal or power-state changes, and unrelated background activity can remain undetected. Review the raw samples and dispersion; the harness does not remove outliers.

`GGML_EXEC_TRACE_INTEGRITY` writes the fixed-size structural witness used by the harness. It does not allocate the raw event buffer or read the host clock at each trace site. It is an internal validation mechanism rather than a user trace format.

## Validation history

The final balanced integrity gate measured -1.51% median decode throughput and -0.01% prefill throughput with DELPHI. A separate balanced decode matrix measured -0.55% for DELPHI, -60.25% for ROCprofiler, and -52.10% for both relative to clean execution. The external-profiler samples had high dispersion, including very slow first samples. Use the integrity harness, without ROCprofiler, to decide whether the in-process trace is production-representative.
