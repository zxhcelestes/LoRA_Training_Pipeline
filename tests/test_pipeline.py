import hashlib
import io
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# Helper
def _make_image(width: int = 512, height: int = 512, mode: str = "RGB") -> Image.Image:
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


def _save_images(directory: Path, count: int = 10, size: tuple = (512, 512)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(count):
        img = _make_image(*size)
        p = directory / f"img_{i:04d}.png"
        img.save(p)
        paths.append(p)
    return paths


# Data Processing

class TestImageValidator:
    def setup_method(self):
        from pipeline.processor import ImageValidator, ProcessingConfig
        self.validator = ImageValidator(ProcessingConfig())

    def test_accepts_valid_png(self, tmp_path):
        p = tmp_path / "ok.png"
        _make_image().save(p)
        valid, reason = self.validator.validate(p)
        assert valid, reason

    def test_rejects_unsupported_extension(self, tmp_path):
        p = tmp_path / "file.bmp2"
        p.write_bytes(b"fake")
        valid, _ = self.validator.validate(p)
        assert not valid

    def test_rejects_too_small(self, tmp_path):
        p = tmp_path / "small.png"
        _make_image(100, 100).save(p)
        valid, reason = self.validator.validate(p)
        assert not valid
        assert "small" in reason.lower() or "too" in reason.lower()

    def test_rejects_corrupt_file(self, tmp_path):
        p = tmp_path / "corrupt.png"
        p.write_bytes(b"this is not an image")
        valid, _ = self.validator.validate(p)
        assert not valid

    def test_blur_score_higher_for_sharp_image(self):
        # Sharp image: high-contrast edges
        sharp = Image.fromarray(np.eye(512, dtype=np.uint8) * 255)
        # Blurry image: uniform noise
        blurry = Image.fromarray(np.ones((512, 512), dtype=np.uint8) * 128)
        s_sharp = self.validator.compute_blur_score(sharp)
        s_blurry = self.validator.compute_blur_score(blurry)
        assert s_sharp > s_blurry

    def test_hash_consistency(self, tmp_path):
        p = tmp_path / "img.png"
        _make_image().save(p)
        h1 = self.validator.compute_hash(p)
        h2 = self.validator.compute_hash(p)
        assert h1 == h2
        assert len(h1) == 32

    def test_duplicate_detection_via_hash(self, tmp_path):
        p = tmp_path / "img.png"
        img = _make_image()
        img.save(p)
        q = tmp_path / "copy.png"
        img.save(q)
        assert self.validator.compute_hash(p) == self.validator.compute_hash(q)


class TestImagePreprocessor:
    def setup_method(self):
        from pipeline.processor import ImagePreprocessor, ProcessingConfig
        self.pp = ImagePreprocessor(ProcessingConfig(target_size=256))

    def test_output_size(self):
        img = _make_image(800, 600)
        out = self.pp.process(img)
        assert out.size == (256, 256)

    def test_output_mode_rgb(self):
        img = _make_image(512, 512).convert("RGBA")
        out = self.pp.process(img)
        assert out.mode == "RGB"

    def test_non_square_input_center_crop(self):
        img = _make_image(800, 400)
        out = self.pp.process(img)
        assert out.size == (256, 256)

    def test_augmentation_returns_correct_count(self):
        from pipeline.processor import ProcessingConfig, ImagePreprocessor
        pp = ImagePreprocessor(ProcessingConfig(augment_factor=3))
        img = _make_image(512, 512)
        augmented = pp.augment(img)
        assert len(augmented) == 3

    def test_augmentation_differs_from_original(self):
        img = _make_image(512, 512)
        augmented = self.pp.augment(img)
        orig_arr = np.array(img)
        for aug in augmented:
            assert not np.array_equal(np.array(aug), orig_arr)


class TestCaptionGenerator:
    def setup_method(self):
        from pipeline.processor import CaptionGenerator, ProcessingConfig
        self.gen = CaptionGenerator(ProcessingConfig(trigger_word="xyz"))

    def test_trigger_word_in_caption(self, tmp_path):
        p = tmp_path / "portrait.png"
        img = _make_image()
        caption = self.gen.generate(p, img)
        assert "xyz" in caption

    def test_filename_context_used(self, tmp_path):
        p = tmp_path / "smiling_outdoors.png"
        img = _make_image()
        caption = self.gen.generate(p, img)
        assert "smiling" in caption or "outdoors" in caption

    def test_generic_fallback(self, tmp_path):
        p = tmp_path / "12345.png"
        img = _make_image()
        caption = self.gen.generate(p, img)
        assert len(caption) > 0


class TestDatasetSplitter:
    def setup_method(self):
        from pipeline.processor import DatasetSplitter, ProcessingConfig, ImageRecord
        self.splitter = DatasetSplitter(ProcessingConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1))
        self.Record = ImageRecord

    def _make_records(self, n: int) -> list:
        return [
            self.Record(
                path=f"/tmp/img_{i}.png",
                caption="test",
                file_hash=f"hash{i}",
            )
            for i in range(n)
        ]

    def test_split_proportions_roughly_correct(self):
        records = self._make_records(100)
        split = self.splitter.split(records)
        train_n = sum(1 for r in split if r.split == "train" and not r.is_augmented)
        val_n = sum(1 for r in split if r.split == "val")
        test_n = sum(1 for r in split if r.split == "test")
        assert train_n >= 75
        assert val_n >= 5
        assert test_n >= 5

    def test_deterministic_with_same_seed(self):
        records_a = self._make_records(50)
        records_b = self._make_records(50)
        a = self.splitter.split(records_a, seed=0)
        b = self.splitter.split(records_b, seed=0)
        assert [r.split for r in a] == [r.split for r in b]

    def test_augmented_images_always_train(self):
        from pipeline.processor import ImageRecord
        records = self._make_records(20)
        for r in records[:5]:
            r.is_augmented = True
        split = self.splitter.split(records)
        for r in split:
            if r.is_augmented:
                assert r.split == "train"


class TestDataProcessor:
    def test_end_to_end_small_dataset(self, tmp_path):
        from pipeline.processor import DataProcessor, ProcessingConfig

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _save_images(input_dir, count=8, size=(512, 512))

        config = ProcessingConfig(target_size=256, augment=True, augment_factor=2)
        processor = DataProcessor(config)
        result = processor.process_dataset(input_dir, output_dir)

        assert result.stats["accepted"] == 8
        assert result.stats["rejected"] == 0
        assert result.stats["augmented"] > 0
        assert result.stats["train"] >= 6

    def test_captions_written_alongside_images(self, tmp_path):
        from pipeline.processor import DataProcessor, ProcessingConfig

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _save_images(input_dir, count=5, size=(512, 512))

        processor = DataProcessor(ProcessingConfig(target_size=256, augment=False))
        result = processor.process_dataset(input_dir, output_dir)

        for record in result.records:
            caption_path = Path(record.path).with_suffix(".txt")
            assert caption_path.exists(), f"Missing caption for {record.path}"
            assert len(caption_path.read_text()) > 0

    def test_rejects_small_images(self, tmp_path):
        from pipeline.processor import DataProcessor, ProcessingConfig

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _save_images(input_dir, count=3, size=(100, 100))

        processor = DataProcessor(ProcessingConfig(target_size=256))
        result = processor.process_dataset(input_dir, output_dir)

        assert result.stats["accepted"] == 0
        assert result.stats["rejected"] == 3

    def test_deduplication_removes_identical_images(self, tmp_path):
        from pipeline.processor import DataProcessor, ProcessingConfig

        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        img = _make_image(512, 512)
        for i in range(4):
            img.save(input_dir / f"dup_{i}.png")

        processor = DataProcessor(ProcessingConfig(target_size=256, augment=False))
        result = processor.process_dataset(input_dir, output_dir)

        assert result.stats["accepted"] == 1


# Training
class TestGPUMemoryManager:
    def test_available_gpus_returns_list(self):
        from pipeline.trainer import GPUMemoryManager
        gpus = GPUMemoryManager.available_gpus()
        assert isinstance(gpus, list)

    def test_clear_cache_no_error(self):
        from pipeline.trainer import GPUMemoryManager
        GPUMemoryManager.clear_cache()

    def test_get_used_gb_cpu_returns_zero(self):
        from pipeline.trainer import GPUMemoryManager
        import torch
        if not torch.cuda.is_available():
            assert GPUMemoryManager.get_used_gb() == 0.0


# Ckpt
class TestCheckpointManager:
    def test_save_and_load_latest(self, tmp_path):
        from pipeline.trainer import CheckpointManager

        cm = CheckpointManager(tmp_path, max_keep=3)
        mock_pipeline = MagicMock()
        mock_pipeline.unet.save_attn_procs = MagicMock()

        import torch
        mock_optimizer = MagicMock()
        mock_optimizer.state_dict.return_value = {}

        with patch("torch.save"):
            cm.save(mock_pipeline, mock_optimizer, 100, {"epoch": 1})
            cm.save(mock_pipeline, mock_optimizer, 200, {"epoch": 2})

        result = cm.load_latest()
        assert result is not None
        path, meta = result
        assert meta["step"] == 200

    def test_prunes_old_checkpoints(self, tmp_path):
        from pipeline.trainer import CheckpointManager

        cm = CheckpointManager(tmp_path, max_keep=2)
        mock_pipeline = MagicMock()
        mock_pipeline.unet.save_attn_procs = MagicMock()
        mock_optimizer = MagicMock()
        mock_optimizer.state_dict.return_value = {}

        with patch("torch.save"):
            for step in [100, 200, 300, 400]:
                cm.save(mock_pipeline, mock_optimizer, step, {"epoch": step // 100})

        remaining = sorted(cm.ckpt_dir.glob("step_*"))
        assert len(remaining) <= 2


class TestLoRATrainer:
    def test_stub_training_returns_result(self, tmp_path):
        from pipeline.trainer import LoRATrainer, LoRAConfig, ImageRecord

        config = LoRAConfig(output_dir=str(tmp_path))
        trainer = LoRATrainer(config)

        records = [
            MagicMock(path=str(tmp_path / f"img_{i}.png"), caption="test", split="train")
            for i in range(5)
        ]

        # Force stub path
        with patch("pipeline.training.trainer.LoRATrainer._train_impl", side_effect=ImportError("no diffusers")):
            result = trainer.train("test_job", records[:4], records[4:])

        assert result.success
        assert result.job_id == "test_job"
        assert result.final_loss > 0
        assert "total_steps" in result.summary

    def test_progress_callback_called(self, tmp_path):
        from pipeline.trainer import LoRATrainer, LoRAConfig

        config = LoRAConfig(output_dir=str(tmp_path))
        trainer = LoRATrainer(config)
        steps_seen = []
        trainer.add_progress_callback(steps_seen.append)

        records = [MagicMock(path=str(tmp_path)) for _ in range(5)]

        with patch("pipeline.training.trainer.LoRATrainer._train_impl", side_effect=ImportError):
            trainer.train("cb_test", records, [])

        assert len(steps_seen) > 0


# Evaluation
class TestCLIPScorer:
    def test_returns_float_in_range(self):
        from pipeline.evaluator import CLIPScorer
        scorer = CLIPScorer(device="cpu")

        with patch.object(scorer, "_load", lambda: None):
            scorer._model = None
            score = scorer.score([_make_image()], ["a test prompt"])

        assert 0.0 <= score <= 1.0

    def test_multiple_images_averaged(self):
        from pipeline.evaluator import CLIPScorer
        scorer = CLIPScorer(device="cpu")
        scorer._model = None
        scores = [
            scorer.score([_make_image()], ["prompt"])
            for _ in range(5)
        ]
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestModelEvaluator:
    def test_evaluate_stub_produces_metrics(self, tmp_path):
        from pipeline.evaluator import ModelEvaluator, EvalConfig

        config = EvalConfig(num_eval_images=3)
        evaluator = ModelEvaluator(config)

        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch("pipeline.evaluation.evaluator.ModelEvaluator._generate_with_diffusers",
                   side_effect=ImportError("no diffusers")):
            metrics = evaluator.evaluate(model_path, output_dir=tmp_path / "eval")

        assert metrics.num_images_generated == 3
        assert 0.0 <= metrics.clip_score <= 1.0
        assert metrics.avg_inference_ms >= 0
        assert (tmp_path / "eval" / "metrics.json").exists()

    def test_passes_threshold_logic(self, tmp_path):
        from pipeline.evaluator import ModelEvaluator, EvalConfig, EvalMetrics

        evaluator = ModelEvaluator(EvalConfig(min_clip_score=0.30))

        with patch.object(evaluator, "_generate_samples", return_value=([_make_image()], [0.1])):
            with patch.object(evaluator.clip_scorer, "score", return_value=0.10):
                metrics = evaluator.evaluate(tmp_path, output_dir=tmp_path / "e1")
        assert not metrics.passes_threshold

        with patch.object(evaluator, "_generate_samples", return_value=([_make_image()], [0.1])):
            with patch.object(evaluator.clip_scorer, "score", return_value=0.35):
                metrics = evaluator.evaluate(tmp_path, output_dir=tmp_path / "e2")
        assert metrics.passes_threshold


class TestABTestFramework:
    def test_winner_selection(self, tmp_path):
        from pipeline.evaluator import (
            ABTestFramework, ModelEvaluator, EvalMetrics
        )
        model_a = tmp_path / "a"
        model_b = tmp_path / "b"
        model_a.mkdir()
        model_b.mkdir()

        evaluator = ModelEvaluator()
        framework = ABTestFramework(evaluator)

        metrics_a = EvalMetrics(clip_score=0.35, fid_score=None, passes_threshold=True)
        metrics_b = EvalMetrics(clip_score=0.25, fid_score=None, passes_threshold=True)

        with patch.object(evaluator, "evaluate", side_effect=[metrics_a, metrics_b]):
            result = framework.run(model_a, model_b, "model_a", "model_b")

        assert result.winner == "A"

    def test_tie_when_difference_small(self, tmp_path):
        from pipeline.evaluator import ABTestFramework, ModelEvaluator, EvalMetrics

        model_a = tmp_path / "a"
        model_b = tmp_path / "b"
        model_a.mkdir()
        model_b.mkdir()

        evaluator = ModelEvaluator()
        framework = ABTestFramework(evaluator)

        clip = 0.30
        m = EvalMetrics(clip_score=clip, fid_score=None, passes_threshold=True)

        with patch.object(evaluator, "evaluate", return_value=m):
            result = framework.run(model_a, model_b)

        assert result.winner == "tie"


# API

class TestAPI:
    def setup_method(self):
        from fastapi.testclient import TestClient
        from api.main import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        r = self.client.get("/health")
        assert r.status_code == 200
        data = r.json()
        assert "status" in data
        assert data["status"] == "ok"

    def test_metrics_endpoint(self):
        r = self.client.get("/metrics")
        assert r.status_code == 200
        assert "lora_jobs_total" in r.text

    def test_create_job_too_few_images(self, tmp_path):
        img = io.BytesIO()
        _make_image().save(img, format="PNG")
        img.seek(0)

        r = self.client.post(
            "/jobs",
            files=[("files", ("img.png", img, "image/png"))],
            data={"config": "{}"},
        )
        assert r.status_code == 422

    def test_get_nonexistent_job(self):
        r = self.client.get("/jobs/doesnotexist")
        assert r.status_code == 404

    def test_list_jobs_empty(self):
        r = self.client.get("/jobs")
        assert r.status_code == 200
        assert "jobs" in r.json()

    def test_cancel_nonexistent_job(self):
        r = self.client.delete("/jobs/ghost")
        assert r.status_code == 404


# Benchmark

def benchmark_data_processing(n_images: int = 50) -> dict:
    """Measure preprocessing throughput."""
    import time
    from pipeline.processor import DataProcessor, ProcessingConfig

    with tempfile.TemporaryDirectory() as tmp:
        input_dir = Path(tmp) / "input"
        output_dir = Path(tmp) / "output"
        _save_images(input_dir, count=n_images, size=(512, 512))

        config = ProcessingConfig(target_size=256, augment=False)
        processor = DataProcessor(config)

        t0 = time.perf_counter()
        result = processor.process_dataset(input_dir, output_dir)
        elapsed = time.perf_counter() - t0

    return {
        "n_images": n_images,
        "processed": result.stats["accepted"],
        "duration_s": round(elapsed, 3),
        "images_per_second": round(n_images / elapsed, 1),
    }


if __name__ == "__main__":
    stats = benchmark_data_processing(20)
    print(json.dumps(stats, indent=2))
