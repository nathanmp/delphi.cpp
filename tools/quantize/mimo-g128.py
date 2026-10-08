#!/usr/bin/env python3
"""Stream MiMo MXFP4 routed experts to uncalibrated G128 and dense weights to Q8_0."""
import argparse
import ctypes
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import re
import sys
import tempfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'gguf-py')]
import gguf
from conversion.base import LazyTorchTensor
from conversion.mimo import MimoV2Model


def unpack_mx(packed, scale):
    rows, half_k = packed.shape
    if scale.shape != (rows, half_k // 16):
        raise ValueError('MXFP4 weight/scale shapes do not match')
    codes = np.stack((packed & 15, packed >> 4), axis=-1).reshape(rows, -1, 32)
    # Use ggml's doubled E2M1 table and half-scale conversion, including subnormal exponents.
    values = np.asarray(gguf.quants.MXFP4.kvalues, np.float32)[codes]
    return np.ascontiguousarray((values * gguf.quants.MXFP4.e8m0_to_fp32_half(scale)[..., None]).reshape(rows, half_k * 2))


class NativeQuantizer:
    def __init__(self, library):
        self.lib = ctypes.CDLL(str(library))
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                               ctypes.c_int64, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.original = gguf.quants.quantize

    def __call__(self, data, qtype):
        if qtype not in (gguf.GGMLQuantizationType.Q4_G128, gguf.GGMLQuantizationType.Q8_0):
            return self.original(data, qtype)
        shape = gguf.quant_shape_to_byte_shape(data.shape, qtype)
        if isinstance(data, gguf.LazyNumpyTensor):
            return gguf.LazyNumpyTensor(meta=gguf.LazyNumpyTensor.meta_with_dtype_and_shape(np.uint8, shape),
                                       args=(data,), func=lambda a: self(a, qtype))
        data = np.ascontiguousarray(data, dtype=np.float32)
        if not np.isfinite(data).all():
            raise ValueError('Nonfinite source weights')
        out = np.empty(shape, np.uint8)
        k = data.shape[-1]
        n = data.size // k
        size = self.lib.ggml_quantize_chunk(int(qtype), data.ctypes.data, out.ctypes.data, 0, n, k, None)
        if size != out.nbytes:
            raise ValueError('Native quantizer returned an unexpected size')
        return out


class ExpertTensor(gguf.LazyChunkedTensor):
    def __init__(self, chunks, shape, quantizer, workers):
        super().__init__(chunks, gguf.quant_shape_to_byte_shape(shape, gguf.GGMLQuantizationType.Q4_G128), np.uint8)
        self.quantizer = quantizer
        self.workers = workers

    def tofile(self, *args, **kwargs):
        # Bound lookahead to the worker count; retain output expert order and one canonical weight copy.
        def convert(load):
            return self.quantizer(load(), gguf.GGMLQuantizationType.Q4_G128)
        chunks = iter(self._chunks)
        written = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            pending = deque(pool.submit(convert, load) for load in [next(chunks) for _ in range(min(self.workers, len(self._chunks)))])
            while pending:
                data = pending.popleft().result()
                data.tofile(*args, **kwargs)
                written += data.nbytes
                del data
                load = next(chunks, None)
                if load is not None:
                    pending.append(pool.submit(convert, load))
        if written != self.nbytes:
            raise ValueError('Unexpected expert bank output size')


class StreamingMimo(MimoV2Model):
    model_arch = gguf.MODEL_ARCH.MIMO2

    def dequant_model(self):
        qc = self.hparams.get('quantization_config', {})
        if qc.get('store_dtype') != 'mxfp4' or qc.get('quant_method') != 'fp8':
            raise ValueError('This tool requires MiMo MOPD MXFP4 experts with FP8 dense weights')
        groups = {}
        pat = re.compile(r'^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$')
        for name in list(self.model_tensors):
            m = pat.match(name)
            if not m:
                continue
            layer, expert, proj = int(m[1]), int(m[2]), m[3]
            scale_name = name + '_scale'
            weight = self.model_tensors.pop(name)
            scale = self.model_tensors.pop(scale_name)
            shape = tuple(weight().shape)
            if weight().dtype != torch.uint8 or scale().dtype != torch.uint8:
                raise ValueError('Expected byte-packed MXFP4 experts and E8M0 scales')
            key = (layer, proj)
            groups.setdefault(key, {})[expert] = (weight, scale, (shape[0], shape[1] * 2))

        n_experts = self.hparams['n_routed_experts']
        expected = sum(bool(x) for x in self.hparams['moe_layer_freq']) * 3
        if len(groups) != expected:
            raise ValueError(f'Expected {expected} routed banks, found {len(groups)}')
        for (layer, proj), experts in sorted(groups.items()):
            if set(experts) != set(range(n_experts)):
                raise ValueError(f'Incomplete expert bank: layer {layer}, {proj}')
            shape = experts[0][2]
            if any(t[2] != shape for t in experts.values()):
                raise ValueError('Expert shapes differ within a bank')
            def loader(w, s):
                def load():
                    return unpack_mx(LazyTorchTensor.to_eager(w()).numpy(), LazyTorchTensor.to_eager(s()).numpy())
                return load
            table = ExpertTensor([loader(experts[e][0], experts[e][1]) for e in range(n_experts)],
                                 (n_experts, *shape), self.native_quantizer, self.expert_workers)
            name = self.map_tensor_name(f'model.layers.{layer}.mlp.experts.{proj}.weight')
            self.gguf_writer.add_tensor(name, table,
                                        raw_dtype=gguf.GGMLQuantizationType.Q4_G128)
        logging.info('Prepared %d G128 routed banks; at most %d experts in flight', len(groups), self.expert_workers)

        # The existing target QKV conversion handles TP4 packing. Apply the same treatment to MTP QKV.
        mtp_qkv = {}
        for name in self.model_tensors:
            if re.match(r'^model\.mtp\.layers\.\d+\.self_attn\.qkv_proj\.weight$', name):
                mtp_qkv[name] = (self.model_tensors[name], self.model_tensors[name + '_scale_inv'])
        super().dequant_model()
        hp = self.hparams
        for name, (w, s) in mtp_qkv.items():
            self.model_tensors[name] = lambda w=w, s=s: self._tp_aware_qkv_dequant(
                w(), s(), hp['num_attention_heads'], hp['swa_num_key_value_heads'], hp['head_dim'], hp['v_head_dim'])


def check(quant):
    codes = np.arange(128, dtype=np.uint8).reshape(4, 32) % 16
    packed = codes[:, ::2] | (codes[:, 1::2] << 4)
    scales = np.array([[0], [1], [120], [127]], np.uint8)
    got = unpack_mx(packed, scales)
    raw = MimoV2Model.repack_mxfp4_blocks(torch.from_numpy(packed), torch.from_numpy(scales))
    ref = gguf.quants.dequantize(raw, gguf.GGMLQuantizationType.MXFP4)
    np.testing.assert_array_equal(got, ref)
    x = np.tile(got[-1], (2, 4))
    assert quant(x, gguf.GGMLQuantizationType.Q4_G128).shape == (2, 66)
    np.testing.assert_array_equal(quant(x, gguf.GGMLQuantizationType.Q8_0), quant.original(x, gguf.GGMLQuantizationType.Q8_0))
    lazy = gguf.LazyNumpyTensor.from_eager(x)
    np.testing.assert_array_equal(gguf.LazyNumpyTensor.to_eager(quant(lazy, gguf.GGMLQuantizationType.Q4_G128)),
                                  quant(x, gguf.GGMLQuantizationType.Q4_G128))
    chunks = [lambda i=i: x * (i + 1) for i in range(11)]
    table = ExpertTensor(chunks, (11, *x.shape), quant, 3)
    with tempfile.TemporaryFile() as out:
        table.tofile(out)
        out.seek(0)
        expected = b''.join(quant(load(), gguf.GGMLQuantizationType.Q4_G128).tobytes() for load in chunks)
        assert out.read() == expected, 'Parallel writer changed expert order or bytes'
    print('MXFP4 nibble/scale and native quantization checks passed', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--libggml', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    quant = NativeQuantizer(args.libggml.resolve())
    check(quant)
    if args.workers < 1:
        p.error('--workers must be positive')
    if args.check_only:
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    partial = args.output.with_suffix('.partial.gguf')
    if partial.exists():
        raise FileExistsError(partial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    gguf.quants.quantize = quant
    with torch.inference_mode():
        model = StreamingMimo(args.model, gguf.LlamaFileType.MOSTLY_Q8_0, partial,
                              model_name='MiMo-V2.6-Flash-MOPD G128 experts Q8_0 dense (uncalibrated)', dry_run=args.dry_run)
        model.native_quantizer = quant
        model.expert_workers = args.workers
        model.write()
    if not args.dry_run:
        partial.rename(args.output)
        print('Completed:', args.output, flush=True)


if __name__ == '__main__':
    main()
