# Vision (image input)

Vision lets a multimodal model read images in chat requests. delphi.cpp runs the vision encoder on a GPU, and it can release that GPU memory between requests. You don't need to set anything beyond the projector file and the device. The default memory mode already keeps the language model's memory free for the encoder.

## Quick start

Pass the projector file with `--mmproj`. The projector is a separate GGUF file made from the same checkpoint as the model:

```sh
llama-server -m glm.gguf --mmproj mmproj.gguf -mmdev CUDA2 -ngl 999 --device CUDA0,CUDA1,CUDA2 -ts 17,14,16 -fa on -c 65536
```

Send an image in the usual chat format. The server encodes the image before it decodes the answer.

If you have no projector yet, `convert_hf_to_gguf.py` produces one with its `--mmproj` option. Run it on the checkpoint.

## Where the encoder runs

By default the encoder weights stay in system RAM. This mode is `dynamic`, and it is the default. Before an image batch, the server releases the language model's compute buffers on the encoder's GPU and uploads the encoder. It encodes the images and then frees the encoder's memory again. The language model rebuilds its buffers at its next decode.

The alternative is `resident`. The encoder stays on the GPU all the time, which is how upstream llama.cpp works:

```sh
llama-server ... --mmproj-memory resident
```

Use `resident` when image latency matters more than the GPU memory it holds. Dynamic mode adds an upload for each batch of images. On a PCIe 2.0 x4 link, the GLM-5.3 Flash encoder (1.1 GB) took about 0.7 s to upload.

Images from one request share one upload. Requests with images run one after the other.

For more detail, including the fitting rules, see [Vision memory](vision-memory.md).

## Choosing the GPU

Use `-mmdev DEVICE` (or `--mmproj-device`) to choose the GPU for the encoder.

On GLM-5.3 Flash with three CMP 170HX cards, `CUDA0` has a PCIe x2 link. Its upload is slower. An image took about 1.9 s with the encoder on `CUDA0` and about 1.2 s on `CUDA2` (measured 2026-10-31). The answers were identical. Using `CUDA2` also left 2.6 GiB more memory free on `CUDA0`.

```sh
llama-server ... --mmproj mmproj.gguf -mmdev CUDA2
```

## DeepSeek V4.1 Flash

DeepSeek V4.1 Flash accepts images with its projector. The expert cache can take the memory that the encoder needs, so raise the cache reserve when the encoder fails to allocate. On 2026-10-23, a 1000-token image failed to allocate a 388 MB compute buffer with the default reserve. A reserve of 6144 MiB worked around the failure:

```sh
llama-server -m deepseek.gguf --mmproj mmproj.gguf --moe-cache-margin 6144 ...
```

Keep the model files on a fast SSD. On 2026-10-23, the model files were on a hard disk and nearly as large as the system RAM. The page cache thrashed, and the decode speed was 1.5 to 6 tokens per second, rising with each request. The test did not measure the speed from an SSD.

## Limits

- The `--fit` option does not count a dynamic projector. Leave free memory on the GPU that holds the encoder.
- Audio encoders are not affected by the memory mode.
- A large image or a long prompt can still fail to allocate. Check the server log for the allocation error, and raise the expert cache reserve if the expert cache is the cause.
- Keep `--parallel 1` with DFlash2. More than one slot has not been tested on long prompts since a fix on 2026-11-01.
- On GLM-5.3 Flash at a 300K context, CUDA2 has about 2 GiB free after a 250K-token prompt. Test a large image at long context before you rely on it.

## Related pages

- [Vision memory](vision-memory.md) describes the memory modes in detail.
- [Expert cache](expert-cache.md) explains the reserve that the expert cache keeps free.
