# exec-trace-gpu-activity and exec-trace-rocm: vendor GPU activity import

Offline GPU activity uses the provider-neutral schema `delphi-gpu-activity-v1`, defined in `tools/exec-trace/exec-trace-gpu-activity.py`. A provider adapter writes one document with these parts:

- `provider`: name, provenance, `production_representative`, timestamp clock, and `graph_node_identity` (`none`, or `provider_reported` when activities carry provider graph IDs that DELPHI does not map to GGML nodes);
- `devices`: provider-scoped `device_ref`, type, GPU index, and names;
- `api_calls`: host API records with thread, stream, correlation IDs, provider timestamps, and a `graph_launch` flag;
- `activities`: `kernel`, `copy`, or `memset` records with device references, queue and stream, provider timestamps, correlation IDs, and copy bytes and direction.

Validation rejects `VENDOR_EXTERNAL` data that claims to be production-representative and graph IDs from a provider that declares none.

## Synopsis

```text
python3 tools/exec-trace/exec-trace-gpu-activity.py validate activity.json
python3 tools/exec-trace/exec-trace-gpu-activity.py merge /tmp/delphi.trace activity.json -o /tmp/merged.json
```

`merge` accepts `--device-map DELPHI_DEVICE_ID=PROVIDER_GPU_INDEX` (repeatable) and `--clock-offset-ns` for providers whose clock the trace cannot anchor. It also accepts `--correlation-tolerance-ns` (default 10000): extra clock-boundary tolerance when matching provider API calls to host spans.

The merge is the same for every provider: clock alignment, device mapping, conservative correlation, and device intervals. A provider whose clock is not `CLOCK_MONOTONIC` needs `--clock-offset-ns` until the trace records an anchor for that clock.

## exec-trace-rocm: ROCprofiler-SDK adapter

`tools/exec-trace/exec-trace-rocm.py` is the ROCprofiler-SDK adapter. ROCprofiler timing is `VENDOR_EXTERNAL`: tracing perturbs the observed run, so use it only for qualitative checks. Use device-event timing and the integrity harness for production-representative measurements.

### Synopsis

```text
python3 tools/exec-trace/exec-trace-rocm.py \
    /tmp/trace/delphi.trace /tmp/trace/rocm_results.json \
    -o /tmp/trace/merged.json --activity-output /tmp/trace/activity.json
```

| flag | meaning |
| --- | --- |
| `-o PATH` | `delphi-gpu-activity-merged-v1` JSON output (required) |
| `--activity-output PATH` | also write the `delphi-gpu-activity-v1` provider document |
| `--pid PID` | ROCprofiler process PID when the JSON contains multiple processes |
| `--device-map DEVICE=GPU` | map a DELPHI device ID to a ROCprofiler GPU index; may be repeated |
| `--clock-offset-ns N` | manual value added to ROCprofiler timestamps; for traces without `CLOCK_SYNC` |
| `--correlation-tolerance-ns N` | extra clock-boundary tolerance for matching HIP APIs to host spans (default 10000) |

### Collection

The following configuration was validated with ROCprofiler-SDK 1.3.2 from ROCm 7.14.0; check `rocprofv3 --version` and the matching [AMD documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.14.0/how-to/using-rocprofv3.html) before reusing it with another version.

The smallest activity-only configuration is:

```text
rocprofv3 --kernel-trace --memory-copy-trace --group-by-queue \
    --output-format json --output-directory /tmp/trace --output-file rocm \
    -- <application> <arguments>
```

Add `--hip-runtime-trace` for DELPHI correlation. HIP API records provide host timestamps and correlation IDs that group kernels or copies under the API operation that submitted them. This does not require `GGML_HIP_OP_TRACE`, ROCTx, graph disabling, or new application synchronization. A complete collection has this form:

```text
GGML_EXEC_TRACE=/tmp/trace/delphi.trace \
rocprofv3 --hip-runtime-trace --kernel-trace --memory-copy-trace --group-by-queue \
    --output-format json --output-directory /tmp/trace --output-file rocm \
    -- numactl --cpunodebind=0 --membind=0 <application> <arguments>
```

Merge the files without rerunning inference with the synopsis command above. `--activity-output` also writes the `delphi-gpu-activity-v1` provider document. The merged output uses schema `delphi-gpu-activity-merged-v1` and keeps normalized records and relationships in separate tables. It copies the provider block. `gpu_activities` contains kernel and copy records with the provider `device_ref`, stream/queue, name, provider (`external_*`) and aligned timestamps, correlation IDs, copy bytes/direction, graph IDs when the provider supplies them, and `provenance`. It references `gpu_api_calls`, `host_submissions`, `host_copies`, `dispatches`, and `gpu_graph_events` by integer ID. `device_intervals` derives kernel-busy, DMA, combined-active, and idle intervals within the first-to-last DELPHI execution window. `host_wait_overlaps` reports how much of each host wait overlaps kernel, DMA, combined GPU activity, or no observed GPU activity.

### Clock and device mapping

ROCprofiler-SDK timestamps on the validated host use `CLOCK_MONOTONIC`. DELPHI schema-v4 traces bracket that clock at startup and finalization. The importer uses both anchors to convert timestamps to the trace's `CLOCK_MONOTONIC_RAW` domain while accounting for clock-rate drift, and it reports the larger bracket sampling uncertainty. Early schema-v4 traces with one anchor use a constant offset. The importer applies a separate 10 us boundary tolerance when it tests whether a HIP API call falls in a host span; this tolerance is not reported as clock uncertainty. Older traces need an explicitly measured `--clock-offset-ns`.

The merge maps a single active GPU automatically. With multiple active GPUs it pairs ordered DELPHI device IDs with ordered provider `gpu_index` values. Use repeatable `--device-map DELPHI_DEVICE_ID=ROCPROFILER_GPU_INDEX` arguments when visibility or device ordering makes that inference unsafe.

ROCprofiler-SDK can place invalid UTF-8 bytes in traced API string arguments. The importer replaces invalid sequences because it does not consume those argument strings, keeps JSON structure validation strict, and reports the replacement count in `diagnostics.rocprofiler_utf8_replacements`.

### Correlation confidence

Cross-trace relationships are conservative:

| relationship | meaning |
| --- | --- |
| `ordered_graph_launch` | provider correlation links activity to a graph-launch API, and that API is the only launch paired with a DELPHI replay in the host submission; the cross-trace pairing remains heuristic |
| `api_within_copy` | a correlated copy API falls within one DELPHI copy-handling span |
| `ordered_within_dispatch` | a correlated API falls within one backend submission, but no GGML node boundary is known |
| `time_window_candidate` | only aligned time, device, and submission order provide a candidate |
| `unattributed` | no safe candidate was found |

ROCprofiler-SDK 1.3.2 does not emit HIP graph execution or graph-node IDs in direct JSON. DELPHI can therefore correlate graph launches and repeated activity groups, but it cannot claim an exact HIP-graph-node-to-GGML-node mapping on this installation. The adapter preserves `graph_exec_id` and `graph_node_id` when a newer JSON producer supplies them and then sets `graph_node_identity` to `provider_reported`, without upgrading the GGML relationship to exact. AMD HIP can also report an in-graph copy as a `__amd_rocclr_copyBuffer` kernel instead of a copy record.

A future CUDA provider (CUPTI activity records or an Nsight Systems export) needs only an adapter that writes `delphi-gpu-activity-v1`: map CUDA devices to `device_ref` and `gpu_index`, set `graph_launch` for `cudaGraphLaunch`, keep CUPTI correlation IDs in `correlation_ids.internal`, and declare its timestamp clock. CUPTI graph node IDs, if collected, are `provider_reported` identity; they do not map to GGML nodes by themselves.
