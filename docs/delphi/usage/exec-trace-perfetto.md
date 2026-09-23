# exec-trace-perfetto: unified timeline and Perfetto export

`tools/exec-trace/exec-trace-perfetto.py` combines the raw DELPHI trace, its device-event timing, and an optional merged GPU activity document. It does not rerun inference or add runtime instrumentation.

## Synopsis

```text
python3 tools/exec-trace/exec-trace-perfetto.py \
    /tmp/trace/delphi.trace \
    --gpu-activity /tmp/trace/merged.json \
    --execution 1 \
    -o /tmp/trace/trace.perfetto.json \
    --unified-output /tmp/trace/unified.json \
    --summary /tmp/trace/summary.json \
    --tree /tmp/trace/tree.md
```

| flag | meaning |
| --- | --- |
| `-o PATH` | Perfetto-compatible JSON output (required) |
| `--gpu-activity PATH` (`--rocm-merged` is an older name) | `delphi-gpu-activity-merged-v1` (or legacy `delphi-rocm-activity-v1`) JSON from `exec-trace-rocm.py` |
| `--execution ID` | export one execution ID instead of the complete execution window |
| `--unified-output PATH` | also write the canonical merged timeline/DAG JSON |
| `--summary PATH` | also write the host/GPU interval summary JSON |
| `--tree PATH` | also write a derived tree-style browsing view |

`--gpu-activity` is optional. The exporter rejects a truncated raw trace and a merged document whose schema or execution intervals do not match the raw trace.

## Viewing

Open `trace.perfetto.json` with the **Open trace file** action at [ui.perfetto.dev](https://ui.perfetto.dev/). The output uses the Chrome Trace Event JSON format accepted by Perfetto. Timestamps in Trace Event records are microseconds relative to the selected execution window so large absolute nanosecond values do not lose precision. Each record also keeps its original aligned nanosecond timestamps in its arguments.

The timeline groups records into these logical processes and tracks:

- host threads, including scheduler submissions, waits, and copy handling;
- CPU backend execution spans;
- scheduler graph, split, GGML node, and cross-backend dependency metadata;
- correlated HIP API calls on the host threads observed by ROCprofiler;
- dispatch and HIP graph lifecycle events for each GPU device;
- `DEVICE_EVENT` submission and operation tracks for each GPU device, placed on the host clock, with status, epoch, and placement uncertainty;
- separate kernel and DMA tracks for each observed queue/stream, labeled with the provider provenance;
- an activity/idle union track for each GPU device.

Node and dispatch events include the available GGML operation, tensor name, output type and shape, backend, implementation and variant. Matmul records also include `M`, `N`, `K`, selected-expert count when applicable, and source-0 weight type and shape. GPU activity keeps its correlation relationship and confidence. When a host submission contains exactly one dispatch, the activity also carries that dispatch as a candidate; otherwise it stays explicitly scoped to the host submission.

Logical flow arrows connect traced producer/copy/destination tensor IDs and connect correlated host API, copy, or submission records to GPU activity. A host copy-handling span is still distinct from an observed GPU DMA record. The former measures host code; the latter measures activity reported by ROCprofiler.

`unified.json` uses schema `delphi-unified-timeline-v1`. It is the canonical offline timeline/DAG representation and contains tracks, overlapping events, flows, the selected window, and the summary. The Perfetto file and optional tree are derived views. `tree.md` groups execution -> graph -> split -> GGML node -> dispatch and then summarizes correlated GPU activity. It is for navigation only and does not represent overlap or a call tree.

The summary distinguishes span sums from interval unions. `timing_provenance` names the class of each section. `gpu_device_event` reports the union of `DEVICE_EVENT` submission and graph-launch intervals, its intersection with CPU compute, status counts, and the largest placement uncertainty. `gpu` reports provider activity with its provider name and provenance. It reports execution wall time, CPU-compute span sum and union, backend-wait sum, copy-wait sum, copy-handling sum, GPU kernel/DMA/combined-active unions, observed idle time, and the intersection of CPU-compute and GPU-active unions. Overall GPU unions merge activity across devices, so simultaneous work on two GPUs counts once; the per-device table retains each device separately. Observed idle means the complement of reported kernel/DMA activity within the selected execution window, not proof that the physical GPU had no other work.

The raw trace currently exposes a synchronous `CPU_COMPUTE` backend span but not individual CPU worker intervals. Perfetto therefore shows a logical CPU backend execution track, not per-worker scheduling.

## End-to-end example

The following produces a small trace that contains one generated token while preserving single-socket placement and normal asynchronous execution:

```text
mkdir -p /tmp/delphi-one-token
GGML_EXEC_TRACE=/tmp/delphi-one-token/delphi.trace \
HIP_VISIBLE_DEVICES=1 \
rocprofv3 --hip-runtime-trace --kernel-trace --memory-copy-trace --group-by-queue \
    --output-format json --output-directory /tmp/delphi-one-token --output-file activity \
    -- numactl --cpunodebind=0 --membind=0 build/bin/llama-cli \
    -m /path/to/model.gguf -p "Trace one token." -n 1 -ngl 999 -cmoe \
    -no-cnv --single-turn </dev/null

python3 tools/exec-trace/exec-trace-rocm.py \
    /tmp/delphi-one-token/delphi.trace \
    /tmp/delphi-one-token/activity_results.json \
    -o /tmp/delphi-one-token/merged.json

python3 tools/exec-trace/exec-trace-dump.py /tmp/delphi-one-token/delphi.trace

python3 tools/exec-trace/exec-trace-perfetto.py \
    /tmp/delphi-one-token/delphi.trace \
    --gpu-activity /tmp/delphi-one-token/merged.json \
    --execution 3 \
    -o /tmp/delphi-one-token/trace.perfetto.json \
    --unified-output /tmp/delphi-one-token/unified.json \
    --summary /tmp/delphi-one-token/summary.json \
    --tree /tmp/delphi-one-token/tree.md
```

The validated command above used execution 3 for the real prompt evaluation after two warm-up executions. Use the decoder output to select the corresponding execution if this changes. A useful reference artifact shows scheduler splits and node/dispatch metadata, CPU and GPU backend work, cross-backend handling, HIP graph lifecycle, actual kernel/DMA activity, observed idle gaps, and the correlation confidence on every host-to-GPU relationship.

The Milestone 1 reference export selected one prompt execution with 934 GGML nodes, 34 scheduler splits, 129 dispatch decisions, 17 HIP graph lifecycle records, 380 GPU kernels, and 32 DMA records. The unified view contained 3,052 events and 520 flow relationships; the compact Perfetto JSON was 1.6 MiB. The raw trace dropped no events. The artifact is not checked in because the ROCprofiler input and normalized intermediate total approximately 94 MiB; use the procedure above to reproduce it.
