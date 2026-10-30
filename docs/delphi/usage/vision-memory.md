# Vision memory

The server enables phase vision by default when the target has an expert cache on the vision GPU and the vision backend supports scratch release (CUDA or HIP). This includes automatic cache tiers that are allocated after model loading. Dense models, disabled caches, other vision devices and unsupported backends retain resident vision. Standalone mtmd callers retain their existing default.

Phase vision keeps encoder weights in host RAM. Before encoding an image batch, the server drains draft work and releases idle language-model and draft compute storage on the vision GPU. The encoder measures the actual batch, allocates transient device weights and workspace, and copies its output embeddings to host storage. It then releases its device resources before language-model decoding resumes. The expert banks and KV cache stay resident.

| variable | effect |
| --- | --- |
| `LLAMA_MMPROJ_PHASE=0` | Disable phase vision and keep the encoder resident. |
| `LLAMA_MMPROJ_PHASE=1` | Force phase vision on CUDA/HIP, including without an expert cache. |
| `LLAMA_MMPROJ_PHASE_GPU_MIB=N` | Cap the available device budget for each encoding. `0` forces CPU encoding. |

When weights and planned workspace do not fit the device budget with 256 MiB of kernel headroom, encoding uses the host weights on the CPU. Device weight or compute-reservation failures also fall back to CPU encoding. Backend handles remain resident. Audio encoder behavior is unchanged.

The capacity saving costs an encoder upload and graph recreation for each encoding. CPU fallback adds image latency. Set `LLAMA_MMPROJ_PHASE=0` when image latency is more important than cache capacity. Kernel headroom is an allowance, not an exact bound on every backend allocation.

Phase vision does not change `LLAMA_MOE_CACHE_AUTO_MARGIN_MIB`; keep the allowance needed for language-model graph growth and other device use. Model fitting remains conservative when the server selects the phase default after constructing the target context. Ordinary language-model prefill staging and cache admission can still change expert membership independently of vision.

Validation covers repeated text/image turns, multiple images, normal MTP decoding, CPU fallback, resource release and per-backend embedding parity on HIP. CUDA builds compile; DeepSeek/DSpark and other vision architectures still require deployment validation.
