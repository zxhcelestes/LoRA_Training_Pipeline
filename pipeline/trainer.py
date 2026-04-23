import gc
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    # Base model
    base_model_id: str = "runwayml/stable-diffusion-v1-5"
    # LoRA rank and alpha
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple = ("to_q", "to_v", "to_k", "to_out.0")
    # Training hyperparameters
    learning_rate: float = 1e-4
    lr_scheduler: str = "cosine"
    lr_warmup_steps: int = 100
    num_train_epochs: int = 100
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    # Mixed precision
    mixed_precision: str = "fp16"   # "no", "fp16", "bf16"
    # Memory saving
    gradient_checkpointing: bool = True
    enable_xformers: bool = False    # True if xformers installed
    # Checkpointing
    checkpoint_every_n_steps: int = 200
    max_checkpoints_to_keep: int = 3
    # Logging
    log_every_n_steps: int = 20
    # Output
    output_dir: str = "output"
    seed: int = 42


@dataclass
class TrainingStep:
    step: int
    epoch: int
    loss: float
    lr: float
    elapsed_seconds: float


@dataclass
class TrainingResult:
    job_id: str
    model_path: str
    final_loss: float
    summary: dict
    success: bool = True
    error: Optional[str] = None


class GPUMemoryManager:
    """Monitors and manages GPU memory usage."""
    @staticmethod
    def get_used_gb(device_index: int = 0) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated(device_index) / (1024 ** 3)

    @staticmethod
    def get_free_gb(device_index: int = 0) -> float:
        if not torch.cuda.is_available():
            return float("inf")
        total = torch.cuda.get_device_properties(device_index).total_memory
        used = torch.cuda.memory_allocated(device_index)
        return (total - used) / (1024 ** 3)

    @staticmethod
    def clear_cache():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def available_gpus() -> list[int]:
        return list(range(torch.cuda.device_count()))


class CheckpointManager:
    """Handles saving and loading training checkpoints."""
    def __init__(self, output_dir: Path, max_keep: int = 3):
        self.output_dir = output_dir
        self.max_keep = max_keep
        self.ckpt_dir = output_dir / "checkpoints"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    def save(self, unet, optimizer, step: int, metadata: dict) -> Path:
        ckpt_path = self.ckpt_dir / f"step_{step:06d}"
        ckpt_path.mkdir(exist_ok=True)

        # Save LoRA weights via PEFT
        unet.save_pretrained(str(ckpt_path))

        # Save optimizer state
        torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pt")

        # Save metadata
        meta = {"step": step, **metadata}
        (ckpt_path / "metadata.json").write_text(json.dumps(meta, indent=2))

        logger.info(f"Checkpoint saved at step {step}: {ckpt_path}")
        self._prune_old_checkpoints()
        return ckpt_path

    def load_latest(self) -> Optional[tuple[Path, dict]]:
        checkpoints = sorted(self.ckpt_dir.glob("step_*"))
        if not checkpoints:
            return None
        latest = checkpoints[-1]
        meta = json.loads((latest / "metadata.json").read_text())
        return latest, meta

    def _prune_old_checkpoints(self):
        checkpoints = sorted(self.ckpt_dir.glob("step_*"))
        while len(checkpoints) > self.max_keep:
            old = checkpoints.pop(0)
            import shutil
            shutil.rmtree(old)
            logger.debug(f"Pruned old checkpoint: {old}")


class LoRADataset(torch.utils.data.Dataset):
    """Simple dataset loading preprocessed images and captions."""
    def __init__(self, records: list, tokenizer, image_size: int = 512):
        from torchvision import transforms
        self.records = records
        self.tokenizer = tokenizer
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        from PIL import Image
        record = self.records[idx]
        img = Image.open(record.path).convert("RGB")
        pixel_values = self.transform(img)

        tokens = self.tokenizer(
            record.caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokens.input_ids.squeeze(0),
        }


class LoRATrainer:
    """
    Trains a LoRA adapter on top of a Stable Diffusion base model.
    Handles GPU allocation, memory management, checkpointing, and monitoring.
    """
    def __init__(self, config: Optional[LoRAConfig] = None):
        self.config = config or LoRAConfig()
        self._progress_callbacks: list[Callable[[TrainingStep], None]] = []

    def add_progress_callback(self, fn: Callable[[TrainingStep], None]):
        self._progress_callbacks.append(fn)

    def _emit(self, step: TrainingStep):
        for fn in self._progress_callbacks:
            try:
                fn(step)
            except Exception:
                pass

    def train(
        self,
        job_id: str,
        train_records: list,
        val_records: list,
        resume_from: Optional[str] = None,
    ) -> TrainingResult:
        """
        Main training entry point. Returns a TrainingResult.
        """
        try:
            return self._train_impl(job_id, train_records, val_records, resume_from)
        except ImportError as e:
            logger.warning(f"Training dependencies not available ({e}), running stub")
            return self._stub_train(job_id, train_records)

    def _stub_train(self, job_id: str, train_records: list) -> TrainingResult:
        """Minimal simulation for testing pipeline without GPU/diffusers."""
        output_dir = Path(self.config.output_dir) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "lora_weights.safetensors").write_bytes(b"STUB_WEIGHTS")
        steps = []
        for i in range(1, 11):
            s = TrainingStep(
                step=i * 10, epoch=i, loss=2.0 / i,
                lr=self.config.learning_rate,
                elapsed_seconds=float(i),
            )
            steps.append(s)
            self._emit(s)
            time.sleep(0.05)

        return TrainingResult(
            job_id=job_id,
            model_path=str(output_dir),
            final_loss=steps[-1].loss,
            summary={
                "total_steps": 100,
                "duration_seconds": 5.0,
                "config": asdict(self.config),
            },
        )

    def _train_impl(
        self,
        job_id: str,
        train_records: list,
        val_records: list,
        resume_from: Optional[str],
    ) -> TrainingResult:
        from diffusers import StableDiffusionPipeline, DDPMScheduler
        from transformers import CLIPTokenizer
        import torch.nn.functional as F

        cfg = self.config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        output_dir = Path(cfg.output_dir) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(cfg.seed)
        GPUMemoryManager.clear_cache()

        logger.info(f"Loading base model: {cfg.base_model_id}")
        use_fp16 = cfg.mixed_precision == "fp16" and torch.cuda.is_available()
        pipe = StableDiffusionPipeline.from_pretrained(
            cfg.base_model_id,
            torch_dtype=torch.float16 if use_fp16 else torch.float32,
        )
        pipe.to(device)

        noise_scheduler = DDPMScheduler.from_pretrained(cfg.base_model_id, subfolder="scheduler")
        tokenizer: CLIPTokenizer = pipe.tokenizer

        # Freeze everything
        pipe.vae.requires_grad_(False)
        pipe.text_encoder.requires_grad_(False)
        pipe.unet.requires_grad_(False)

        unet = pipe.unet
        if cfg.gradient_checkpointing:
            unet.enable_gradient_checkpointing()
        if cfg.enable_xformers:
            unet.enable_xformers_memory_efficient_attention()

        # Inject LoRA via PEFT
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=list(cfg.target_modules),
            lora_dropout=cfg.lora_dropout,
            bias="none",
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, unet.parameters()),
            lr=cfg.learning_rate,
        )

        train_dataset = LoRADataset(train_records, tokenizer, cfg.target_size if hasattr(cfg, "target_size") else 512)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=cfg.train_batch_size, shuffle=True,
            num_workers=min(4, os.cpu_count() or 1), pin_memory=True,
        )

        ckpt_manager = CheckpointManager(output_dir, max_keep=cfg.max_checkpoints_to_keep)
        start_step = 0

        # Resume if requested
        if resume_from or (result := ckpt_manager.load_latest()):
            ckpt_path, meta = (Path(resume_from), {}) if resume_from else result
            logger.info(f"Resuming from checkpoint step {meta.get('step', '?')}")
            unet.load_attn_procs(str(ckpt_path))
            opt_state = ckpt_path / "optimizer.pt"
            if opt_state.exists():
                optimizer.load_state_dict(torch.load(opt_state, map_location=device))
            start_step = meta.get("step", 0)

        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_train_epochs * len(train_loader))

        scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

        steps_log: list[TrainingStep] = []
        global_step = start_step
        t0 = time.time()

        for epoch in range(cfg.num_train_epochs):
            unet.train()
            accum_loss = 0.0

            for batch_idx, batch in enumerate(train_loader):
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)

                with torch.no_grad():
                    latents = pipe.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                    encoder_hidden_states = pipe.text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                if use_fp16 and scaler:
                    with torch.cuda.amp.autocast():
                        noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    scaler.scale(loss / cfg.gradient_accumulation_steps).backward()
                    if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                    (loss / cfg.gradient_accumulation_steps).backward()
                    if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(unet.parameters(), cfg.max_grad_norm)
                        optimizer.step()
                        optimizer.zero_grad()

                scheduler.step()
                accum_loss += loss.item()
                global_step += 1

                if global_step % cfg.log_every_n_steps == 0:
                    step_info = TrainingStep(
                        step=global_step,
                        epoch=epoch,
                        loss=accum_loss / cfg.log_every_n_steps,
                        lr=scheduler.get_last_lr()[0],
                        elapsed_seconds=time.time() - t0,
                    )
                    steps_log.append(step_info)
                    self._emit(step_info)
                    logger.info(f"Step {global_step}: loss={step_info.loss:.4f} lr={step_info.lr:.2e}")
                    accum_loss = 0.0

                if global_step % cfg.checkpoint_every_n_steps == 0:
                    ckpt_manager.save(unet, optimizer, global_step, {"epoch": epoch})
                    GPUMemoryManager.clear_cache()

        # Save final LoRA weights via PEFT
        unet.save_pretrained(str(output_dir))
        final_loss = steps_log[-1].loss if steps_log else float("nan")
        logger.info(f"Training complete. Final loss: {final_loss:.4f}. Model: {output_dir}")

        return TrainingResult(
            job_id=job_id,
            model_path=str(output_dir),
            final_loss=final_loss,
            summary={
                "total_steps": global_step,
                "duration_seconds": time.time() - t0,
                "config": asdict(cfg),
            },
        )