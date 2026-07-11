import base64
import json
import tempfile
import unittest
from pathlib import Path

from tools.embedding_compat.bailian_qwen_embed import build_bailian_payload, extract_embedding
from tools.embedding_compat.io_utils import image_to_data_uri, load_manifest, make_result
from tools.embedding_compat.local_qwen_embed import build_local_inputs


class ManifestContractTests(unittest.TestCase):
    def test_load_manifest_resolves_relative_image_and_preserves_text(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "frame.jpg").write_bytes(b"jpeg")
            manifest = root / "inputs.jsonl"
            manifest.write_text(
                "\n".join([
                    json.dumps({"id": "frame-1", "image": "frame.jpg"}),
                    json.dumps({"id": "query-1", "text": "海边日落"}),
                ]),
                encoding="utf-8",
            )

            records = load_manifest(manifest)

            self.assertEqual(records[0]["image"], str((root / "frame.jpg").resolve()))
            self.assertEqual(records[1], {"id": "query-1", "text": "海边日落"})

    def test_load_manifest_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "inputs.jsonl"
            manifest.write_text(
                '{"id":"same","text":"a"}\n{"id":"same","text":"b"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate id"):
                load_manifest(manifest)

    def test_load_manifest_requires_exactly_one_modality(self):
        with tempfile.TemporaryDirectory() as td:
            manifest = Path(td) / "inputs.jsonl"
            manifest.write_text(
                '{"id":"bad","text":"a","image":"b.jpg"}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "exactly one"):
                load_manifest(manifest)

    def test_image_to_data_uri_detects_jpeg_and_png(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            jpg = root / "a.jpg"
            png = root / "b.png"
            jpg.write_bytes(b"jpg-data")
            png.write_bytes(b"png-data")

            self.assertEqual(
                image_to_data_uri(jpg),
                "data:image/jpeg;base64," + base64.b64encode(b"jpg-data").decode("ascii"),
            )
            self.assertEqual(
                image_to_data_uri(png),
                "data:image/png;base64," + base64.b64encode(b"png-data").decode("ascii"),
            )

    def test_make_result_has_stable_cross_runner_schema(self):
        row = make_result(
            record={"id": "q1", "text": "海边"},
            model="model-a",
            dimension=2,
            instruction="retrieve images",
            embedding=[0.1, 0.2],
        )

        self.assertEqual(row["schema_version"], 1)
        self.assertEqual(row["id"], "q1")
        self.assertEqual(row["modality"], "text")
        self.assertEqual(row["embedding"], [0.1, 0.2])
        self.assertEqual(row["model"], "model-a")
        self.assertEqual(row["dimension"], 2)
        self.assertEqual(row["instruction"], "retrieve images")


class BailianContractTests(unittest.TestCase):
    def test_build_text_payload_uses_independent_embedding_and_shared_settings(self):
        payload = build_bailian_payload(
            {"id": "q1", "text": "海边日落"},
            model="qwen3-vl-embedding",
            dimension=2048,
            instruction="Retrieve relevant video frames.",
        )

        self.assertEqual(payload["model"], "qwen3-vl-embedding")
        self.assertEqual(payload["input"], {"contents": [{"text": "海边日落"}]})
        self.assertEqual(payload["parameters"]["dimension"], 2048)
        self.assertEqual(payload["parameters"]["instruct"], "Retrieve relevant video frames.")
        self.assertFalse(payload["parameters"]["enable_fusion"])

    def test_build_image_payload_embeds_local_file_as_data_uri(self):
        with tempfile.TemporaryDirectory() as td:
            image = Path(td) / "frame.jpg"
            image.write_bytes(b"frame")

            payload = build_bailian_payload(
                {"id": "f1", "image": str(image)},
                model="qwen3-vl-embedding",
                dimension=2048,
                instruction="Retrieve relevant video frames.",
            )

            value = payload["input"]["contents"][0]["image"]
            self.assertTrue(value.startswith("data:image/jpeg;base64,"))

    def test_extract_embedding_accepts_model_studio_response(self):
        body = {"output": {"embeddings": [{"index": 0, "type": "vl", "embedding": [3, 4]}]}}

        self.assertEqual(extract_embedding(body), [3.0, 4.0])

    def test_extract_embedding_rejects_missing_vector(self):
        with self.assertRaisesRegex(ValueError, "embedding vector"):
            extract_embedding({"output": {"embeddings": []}})


class LocalQwenContractTests(unittest.TestCase):
    def test_build_local_inputs_uses_same_instruction_for_text_and_image(self):
        records = [
            {"id": "q1", "text": "海边日落"},
            {"id": "f1", "image": "/frames/one.jpg"},
        ]

        inputs = build_local_inputs(records, "Retrieve relevant video frames.")

        self.assertEqual(inputs, [
            {"text": "海边日落", "instruction": "Retrieve relevant video frames."},
            {"image": "/frames/one.jpg", "instruction": "Retrieve relevant video frames."},
        ])

    def test_build_local_inputs_does_not_leak_manifest_id_into_model_input(self):
        inputs = build_local_inputs([{"id": "q1", "text": "test"}], "Represent input.")

        self.assertNotIn("id", inputs[0])


if __name__ == "__main__":
    unittest.main()
