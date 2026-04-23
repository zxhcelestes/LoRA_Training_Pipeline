import asyncio
import json
import logging
import os
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from enum import Enum
from pathlib import Path
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from pipeline.processor import DataProcessor, ProcessingConfig
from pipeline.trainer import LoRATrainer, LoRAConfig, TrainingStep
from pipeline.evaluator import ModelEvaluator, EvalConfig, ABTestFramework

logger = logging.getLogger(__name__)

# Job state
class JobStatus(str, Enum):
    PENDING   = "pending"
    PROCESSING = "processing"
    TRAINING  = "training"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED    = "failed"


class JobRecord:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status: JobStatus = JobStatus.PENDING
        self.progress: dict = {}
        self.result: dict = {}
        self.error: Optional[str] = None
        self.steps: list[dict] = []       # training step logs
        self._step_queue: asyncio.Queue = asyncio.Queue()

    def push_step(self, step: TrainingStep):
        self.steps.append(asdict(step))
        try:
            self._step_queue.put_nowait(asdict(step))
        except asyncio.QueueFull:
            pass


_jobs: dict[str, JobRecord] = {}

# Lifespan & app
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("LoRA Pipeline API starting up")
    yield
    logger.info("LoRA Pipeline API shutting down")


app = FastAPI(
    title="LoRA Training Pipeline API",
    version="1.0.0",
    description="Automatic LoRA training pipeline for image generation models",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class TrainingRequest(BaseModel):
    trigger_word: str = Field("sks", description="Trigger word for the LoRA concept")
    base_model_id: str = Field("runwayml/stable-diffusion-v1-5")
    lora_rank: int = Field(16, ge=1, le=128)
    learning_rate: float = Field(1e-4, gt=0)
    num_epochs: int = Field(100, ge=1, le=2000)
    target_size: int = Field(512, ge=256, le=1024)


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: dict
    result: Optional[dict] = None
    error: Optional[str] = None


class EvalRequest(BaseModel):
    model_path: str
    real_image_dir: Optional[str] = None


class ABTestRequest(BaseModel):
    model_a_path: str
    model_b_path: str
    model_a_id: str = "model_a"
    model_b_id: str = "model_b"
    real_image_dir: Optional[str] = None


# Background task
def _run_pipeline(
    job: JobRecord,
    upload_dir: str,
    request: TrainingRequest,
):
    """Full pipeline: preprocess → train → evaluate. Runs in a thread pool."""
    output_root = Path("outputs") / job.job_id

    try:
        # Data processing
        job.status = JobStatus.PROCESSING
        job.progress = {"stage": "data_processing", "pct": 0}
        logger.info(f"[{job.job_id}] Starting data processing")

        proc_config = ProcessingConfig(
            target_size=request.target_size,
            trigger_word=request.trigger_word,
        )
        processor = DataProcessor(proc_config)
        proc_result = processor.process_dataset(upload_dir, output_root / "dataset")

        if len(proc_result.records) < 5:
            raise ValueError(
                f"Too few valid images after processing: {len(proc_result.records)}. "
                f"Need at least 5."
            )

        job.progress = {
            "stage": "data_processing",
            "pct": 100,
            "stats": proc_result.stats,
        }

        # Training
        job.status = JobStatus.TRAINING
        job.progress["stage"] = "training"
        logger.info(f"[{job.job_id}] Starting training")

        train_records = [r for r in proc_result.records if r.split == "train"]
        val_records   = [r for r in proc_result.records if r.split == "val"]

        lora_config = LoRAConfig(
            base_model_id=request.base_model_id,
            lora_rank=request.lora_rank,
            lora_alpha=request.lora_rank * 2,       # always 2× rank
            learning_rate=request.learning_rate,
            num_train_epochs=request.num_epochs,
            train_batch_size=1,                      # system-managed
            output_dir=str(output_root / "models"),
        )
        trainer = LoRATrainer(lora_config)
        trainer.add_progress_callback(job.push_step)

        train_result = trainer.train(job.job_id, train_records, val_records)
        if not train_result.success:
            raise RuntimeError(f"Training failed: {train_result.error}")

        job.progress["model_path"] = train_result.model_path

        # Evaluation
        job.status = JobStatus.EVALUATING
        job.progress["stage"] = "evaluating"
        logger.info(f"[{job.job_id}] Starting evaluation")

        evaluator = ModelEvaluator(EvalConfig())
        real_dir = output_root / "dataset"
        metrics = evaluator.evaluate(train_result.model_path, real_dir)

        job.status = JobStatus.COMPLETED
        job.result = {
            "model_path": train_result.model_path,
            "final_loss": train_result.final_loss,
            "summary": train_result.summary,
            "metrics": asdict(metrics),
            "passes_quality_threshold": metrics.passes_threshold,
            "data_stats": proc_result.stats,
        }
        job.progress = {"stage": "completed", "pct": 100}
        logger.info(f"[{job.job_id}] Pipeline complete. passes={metrics.passes_threshold}")

    except Exception as exc:
        logger.exception(f"[{job.job_id}] Pipeline failed: {exc}")
        job.status = JobStatus.FAILED
        job.error = str(exc)
    finally:
        # Clean up upload dir
        shutil.rmtree(upload_dir, ignore_errors=True)
        # Signal SSE consumers that the stream is done
        job._step_queue.put_nowait(None)


# Endpoints
@app.get("/health", tags=["System"])
async def health():
    import torch
    return {
        "status": "ok",
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
        "active_jobs": sum(
            1 for j in _jobs.values()
            if j.status in (JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.TRAINING, JobStatus.EVALUATING)
        ),
    }


@app.get("/metrics", tags=["System"])
async def prometheus_metrics():
    """Minimal Prometheus-compatible text metrics."""
    active = sum(
        1 for j in _jobs.values()
        if j.status in (JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.TRAINING, JobStatus.EVALUATING)
    )
    total = len(_jobs)
    completed = sum(1 for j in _jobs.values() if j.status == JobStatus.COMPLETED)
    failed = sum(1 for j in _jobs.values() if j.status == JobStatus.FAILED)
    text = (
        f"# HELP lora_jobs_total Total number of training jobs\n"
        f"# TYPE lora_jobs_total counter\n"
        f"lora_jobs_total {total}\n"
        f"lora_jobs_active {active}\n"
        f"lora_jobs_completed {completed}\n"
        f"lora_jobs_failed {failed}\n"
    )
    return StreamingResponse(iter([text]), media_type="text/plain")


@app.post("/jobs", tags=["Training"], status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(..., description="Image files (100–1000)"),
    config: str = Form("{}", description="JSON-encoded TrainingRequest fields"),
):
    """
    Upload images and start a LoRA training job.
    Returns a job_id for polling or SSE streaming.
    """
    if len(files) < 5:
        raise HTTPException(422, "Provide at least 5 images")
    if len(files) > 1000:
        raise HTTPException(422, "Maximum 1000 images per job")

    try:
        req = TrainingRequest(**json.loads(config))
    except Exception as e:
        raise HTTPException(422, f"Invalid config: {e}")

    job_id = uuid.uuid4().hex
    job = JobRecord(job_id)
    _jobs[job_id] = job

    # Save uploads to temp dir
    upload_dir = tempfile.mkdtemp(prefix=f"lora_{job_id}_")
    for f in files:
        dest = Path(upload_dir) / f.filename
        content = await f.read()
        dest.write_bytes(content)

    background_tasks.add_task(_run_pipeline, job, upload_dir, req)
    return {"job_id": job_id, "status": job.status}


@app.get("/jobs/{job_id}", tags=["Training"], response_model=JobStatusResponse)
async def get_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return JobStatusResponse(
        job_id=job_id,
        status=job.status,
        progress=job.progress,
        result=job.result or None,
        error=job.error,
    )


@app.get("/jobs", tags=["Training"])
async def list_jobs(limit: int = 20, offset: int = 0):
    items = list(_jobs.values())
    items.sort(key=lambda j: j.job_id, reverse=True)
    page = items[offset: offset + limit]
    return {
        "total": len(items),
        "jobs": [
            {"job_id": j.job_id, "status": j.status, "error": j.error}
            for j in page
        ],
    }


@app.delete("/jobs/{job_id}", tags=["Training"])
async def cancel_job(job_id: str):
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        raise HTTPException(409, "Job already finished")
    job.status = JobStatus.FAILED
    job.error = "Cancelled by user"
    return {"job_id": job_id, "status": job.status}


@app.get("/jobs/{job_id}/progress", tags=["Training"])
async def stream_progress(job_id: str):
    """Server-Sent Events stream of training step updates."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        while True:
            try:
                item = await asyncio.wait_for(job._step_queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                yield "event: ping\ndata: {}\n\n"
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
                    break
                continue

            if item is None:                       # sentinel — stream finished
                final = {"status": job.status, "result": job.result, "error": job.error}
                yield f"event: done\ndata: {json.dumps(final)}\n\n"
                break

            yield f"event: step\ndata: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/evaluate", tags=["Evaluation"])
async def evaluate_model(req: EvalRequest):
    if not Path(req.model_path).exists():
        raise HTTPException(404, "Model path not found")
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(req.model_path, req.real_image_dir)
    return asdict(metrics)


@app.post("/ab-test", tags=["Evaluation"])
async def ab_test(req: ABTestRequest):
    framework = ABTestFramework()
    result = framework.run(
        model_a_path=req.model_a_path,
        model_b_path=req.model_b_path,
        model_a_id=req.model_a_id,
        model_b_id=req.model_b_id,
        real_image_dir=req.real_image_dir,
    )
    return asdict(result)


# Dev entrypoint
if __name__ == "__main__":
    import uvicorn
    logging.basicConfig(level=logging.INFO)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
