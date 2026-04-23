import os
import hashlib
import logging
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MIN_SIZE = 256
MAX_SIZE = 4096
MAX_FILE_MB = 20


@dataclass
class ProcessingConfig:
    target_size: int = 512
    min_size: int = MIN_SIZE
    max_file_mb: float = MAX_FILE_MB
    augment: bool = True
    augment_factor: int = 2 # how many augmented copies per image
    train_ratio: float = 0.85
    val_ratio: float = 0.10
    test_ratio: float = 0.05
    caption_template: str = "a photo of {trigger}"
    trigger_word: str = "sks"
    blur_threshold: float = 100.0 # Laplacian variance threshold for blur detection


@dataclass
class ImageRecord:
    path: str
    caption: str
    file_hash: str
    split: str = "train"
    is_augmented: bool = False


@dataclass
class ProcessingResult:
    records: list[ImageRecord] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


class ImageValidator:
    """Validates images for format, size, integrity, and quality."""
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def validate(self, image_path: Path) -> tuple[bool, str]:
        """Returns (is_valid, reason). reason is empty string on success."""
        if image_path.suffix.lower() not in SUPPORTED_FORMATS:
            return False, f"Unsupported format: {image_path.suffix}"

        size_mb = image_path.stat().st_size / (1024 * 1024)
        if size_mb > self.config.max_file_mb:
            return False, f"File too large: {size_mb:.1f}MB"

        try:
            img = Image.open(image_path)
            img.verify()
        except (UnidentifiedImageError, Exception) as e:
            return False, f"Corrupt image: {e}"

        try:
            img = Image.open(image_path)
            w, h = img.size
        except Exception as e:
            return False, f"Cannot read dimensions: {e}"

        if w < self.config.min_size or h < self.config.min_size:
            return False, f"Too small: {w}x{h} (min {self.config.min_size})"

        return True, ""

    def compute_blur_score(self, img: Image.Image) -> float:
        """Returns Laplacian variance as sharpness proxy. Higher = sharper."""
        gray = np.array(img.convert("L"), dtype=np.float32)
        laplacian = (
            gray[:-2, 1:-1] + gray[2:, 1:-1] +
            gray[1:-1, :-2] + gray[1:-1, 2:] -
            4 * gray[1:-1, 1:-1]
        )
        return float(np.var(laplacian))

    def compute_hash(self, image_path: Path) -> str:
        h = hashlib.md5()
        with open(image_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()


class ImagePreprocessor:
    """Resizes, normalizes, and optionally augments images."""
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def process(self, img: Image.Image) -> Image.Image:
        """Center-crop to square, then resize to target_size."""
        img = img.convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((self.config.target_size, self.config.target_size), Image.LANCZOS)
        return img

    def augment(self, img: Image.Image) -> list[Image.Image]:
        """Returns a list of augmented variants (does not include original)."""
        variants = []
        # Horizontal flip
        variants.append(ImageOps.mirror(img))
        # Slight rotation
        for angle in [-10, 10]:
            rotated = img.rotate(angle, expand=False, fillcolor=(0, 0, 0))
            variants.append(rotated)
        # Brightness jitter (simple via numpy)
        arr = np.array(img, dtype=np.float32)
        factor = random.uniform(0.85, 1.15)
        arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
        variants.append(Image.fromarray(arr))

        return variants[: self.config.augment_factor]


class CaptionGenerator:
    """Generates captions for training images."""
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def generate(self, image_path: Path, img: Image.Image) -> str:
        """
        Production note: swap the template fallback here for a BLIP/CogVLM call
        when a captioning model is available.
        """
        trigger = self.config.trigger_word
        stem = image_path.stem.lower().replace("_", " ").replace("-", " ")

        # Try to extract useful context from the filename
        stop_words = {"img", "image", "photo", "pic", "dsc", "file"}
        words = [w for w in stem.split() if w not in stop_words and not w.isdigit()]

        if words:
            description = " ".join(words[:4])
            return f"{trigger}, {description}, high quality photo"

        return self.config.caption_template.format(trigger=trigger)


class DatasetSplitter:
    """Splits records into train/val/test sets deterministically."""
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def split(self, records: list[ImageRecord], seed: int = 42) -> list[ImageRecord]:
        # Only split non-augmented originals, augmented copies follow their parent
        originals = [r for r in records if not r.is_augmented]
        augmented = [r for r in records if r.is_augmented]

        rng = random.Random(seed)
        rng.shuffle(originals)

        n = len(originals)
        n_train = int(n * self.config.train_ratio)
        n_val = int(n * self.config.val_ratio)

        for i, r in enumerate(originals):
            if i < n_train:
                r.split = "train"
            elif i < n_train + n_val:
                r.split = "val"
            else:
                r.split = "test"

        # Augmented images always go to train
        for r in augmented:
            r.split = "train"

        return originals + augmented


class DataProcessor:
    """Orchestrates the full data processing pipeline."""

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self.validator = ImageValidator(self.config)
        self.preprocessor = ImagePreprocessor(self.config)
        self.captioner = CaptionGenerator(self.config)
        self.splitter = DatasetSplitter(self.config)

    def process_dataset(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> ProcessingResult:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result = ProcessingResult()
        seen_hashes: set[str] = set()

        image_paths = [
            p for p in input_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_FORMATS
        ]
        logger.info(f"Found {len(image_paths)} candidate images in {input_dir}")

        for img_path in image_paths:
            valid, reason = self.validator.validate(img_path)
            if not valid:
                result.rejected.append({"path": str(img_path), "reason": reason})
                logger.debug(f"Rejected {img_path.name}: {reason}")
                continue

            # Deduplication
            file_hash = self.validator.compute_hash(img_path)
            if file_hash in seen_hashes:
                result.rejected.append({"path": str(img_path), "reason": "duplicate"})
                continue
            seen_hashes.add(file_hash)

            img = Image.open(img_path).convert("RGB")
            blur_score = self.validator.compute_blur_score(img)
            if blur_score < self.config.blur_threshold:
                result.rejected.append({"path": str(img_path), "reason": f"too blurry ({blur_score:.1f})"})
                continue

            # Preprocess
            processed = self.preprocessor.process(img)
            out_name = f"{file_hash[:12]}.png"
            out_path = output_dir / out_name
            processed.save(out_path, format="PNG")

            caption = self.captioner.generate(img_path, processed)

            record = ImageRecord(
                path=str(out_path),
                caption=caption,
                file_hash=file_hash,
            )
            result.records.append(record)

            # Augmentation
            if self.config.augment:
                for i, aug_img in enumerate(self.preprocessor.augment(processed)):
                    aug_name = f"{file_hash[:12]}_aug{i}.png"
                    aug_path = output_dir / aug_name
                    aug_img.save(aug_path, format="PNG")
                    result.records.append(ImageRecord(
                        path=str(aug_path),
                        caption=caption,
                        file_hash=f"{file_hash}_aug{i}",
                        is_augmented=True,
                    ))

        result.records = self.splitter.split(result.records)
        self._write_captions(result.records, output_dir)

        result.stats = {
            "total_input": len(image_paths),
            "accepted": len([r for r in result.records if not r.is_augmented]),
            "augmented": len([r for r in result.records if r.is_augmented]),
            "rejected": len(result.rejected),
            "train": len([r for r in result.records if r.split == "train"]),
            "val": len([r for r in result.records if r.split == "val"]),
            "test": len([r for r in result.records if r.split == "test"]),
        }
        logger.info(f"Processing complete: {result.stats}")
        return result

    def _write_captions(self, records: list[ImageRecord], output_dir: Path) -> None:
        """Write caption .txt files alongside each image (diffusers convention)."""
        for record in records:
            caption_path = Path(record.path).with_suffix(".txt")
            caption_path.write_text(record.caption, encoding="utf-8")
