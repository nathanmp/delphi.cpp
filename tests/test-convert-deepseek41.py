#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "gguf-py"))

try:
    from conversion.deepseek import DeepseekV41Model
except ModuleNotFoundError as exc:
    print(f"SKIP: converter dependency is unavailable: {exc}")
    sys.exit(77)


class RecordingWriter:
    def __init__(self):
        self.candidate_blocks = None

    def add_candidate_blocks(self, source_layer, block_size, topk_blocks):
        self.candidate_blocks = (source_layer, block_size, topk_blocks)


def make_model(**overrides):
    model = object.__new__(DeepseekV41Model)
    model.hparams = {
        "num_hidden_layers": 40,
        "index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36],
        "candidate_source_layer_id": 20,
        "candidate_block_size": 8,
        "candidate_topk_blocks": 2048,
    }
    model.hparams.update(overrides)
    model.gguf_writer = RecordingWriter()
    return model


class TestDeepseekV41CandidateMetadata(unittest.TestCase):
    def test_writes_runtime_candidate_contract(self):
        model = make_model()
        model._write_candidate_metadata()
        self.assertEqual(model.gguf_writer.candidate_blocks, (20, 8, 2048))

    def test_rejects_missing_candidate_metadata(self):
        for key in (
            "candidate_source_layer_id",
            "candidate_block_size",
            "candidate_topk_blocks",
        ):
            with self.subTest(key=key):
                model = make_model(**{key: None})
                with self.assertRaisesRegex(ValueError, key):
                    model._write_candidate_metadata()

    def test_rejects_non_index_source(self):
        model = make_model(candidate_source_layer_id=21)
        with self.assertRaisesRegex(ValueError, "not an index source"):
            model._write_candidate_metadata()


if __name__ == "__main__":
    unittest.main()
