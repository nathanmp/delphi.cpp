# Vision memory

`llama-server --mmproj-memory dynamic|resident` (env `LLAMA_ARG_MMPROJ_MEMORY`) sets where the vision encoder weights live. The default is `dynamic`.

- `dynamic`: the weights stay in RAM. Before it encodes an image batch, the server drains the draft and releases the compute buffers of the language model and the draft on the vision GPU. The encoder then measures the batch, uploads its weights, allocates its workspace and copies the embeddings to host memory. It releases all of its device memory before the language model decodes again. The KV cache and the expert cache stay resident.
- `resident`: the weights and the encoder workspace stay on the GPU, as in upstream llama.cpp.

Dynamic vision needs CUDA or HIP (scratch release). On other backends, and with `--no-mmproj-offload`, the encoder is resident or on the CPU as before. The startup log shows `vision memory: dynamic` or `resident`. Audio encoders are not changed.

If the weights and the workspace of a batch do not fit in the free memory with 256 MiB of kernel headroom, or a device allocation fails, that batch is encoded on the CPU from the host weights. `LLAMA_MMPROJ_DYNAMIC_GPU_MIB=N` limits the device memory of each encoding (`0` forces the CPU).

## Costs

- Each encoding uploads the encoder weights (GLM 5.3: 1.1 GB, about 0.7 s on a PCIe 2.0 x4 CMP) and the language model reserves its compute buffers again at the next decode.
- Images of one request are encoded in one batch, so they share one upload. Concurrent requests with images are encoded one after the other in the server loop; each has its own upload. Nothing else changes for requests without images.
- Use `resident` when image latency matters more than VRAM.

## Fitting

- `--fit` in the server does not count a dynamic projector.
- `llama-fit-params --mmproj F --mmproj-memory dynamic` prints `mmproj-dynamic DEV MIB`: the memory that one encoding of the warmup image needs on DEV.
- The auto-quantizer (`--mmproj-memory`, default `dynamic`) reserves only what an encoding needs beyond the language model's compute buffers on that GPU, which is usually 0.
