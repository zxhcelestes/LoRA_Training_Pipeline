# Automatic LoRA Training Pipeline

## System Architecture

![System_structure](./System_Structure.png)

## Technical Specification

### Detailed Component Descriptions

**API Gateway**
The single entry point for all client interactions. Responsible for authenticating requests, validating inputs, routing to downstream services, and returning structured responses. Expose both synchronous REST endpoints and a real-time progress stream. Remain stateless so it can be horizontally scaled independently of the training workers.

**Job Scheduler**
Manage the lifecycle of training jobs under constrained GPU resources. Accept incoming job requests, places them in a priority queue, and dispatch them to available workers. Enforce concurrency limits to prevent GPU memory contention. Responsible for tracking job state transitions: pending $\rightarrow$ processing $\rightarrow$ training $\rightarrow$ evaluating $\rightarrow$ completed / failed.

**Data Processor**
Transforms raw user-uploaded images into a clean, model-ready dataset. Responsible for quality filtering, deduplication, normalization, text annotation, data augmentation, and dataset splitting. Be deterministic given the same inputs and seed, and produce output in a format directly consumable by the training component.

**LoRA Trainer**
Train a LoRA on top of a frozen base diffusion model using the processed dataset. Accept a configurable set of hyperparameters. Support periodic checkpointing so that training can be interrupted and resumed without data loss.

**Evaluator**
Assess the quality of a trained LoRA model by generating sample images and scoring them against defined quality metrics. Compare generated outputs to both the text prompts (semantic alignment) and the original training distribution (visual fidelity). Apply configurable thresholds to determine whether a model meets the minimum quality bar for deployment.

**Checkpoint Manager**
Persist training state at regular intervals so that jobs can recover from interruptions. Manage storage of model weights and optimiser state. Enforce a retention policy to bound disk usage, retaining only the most recent N checkpoints.

**Model Registry**
Store versioned, deployment-ready model artifacts. Each entry is associated with metadata including the job that produced it, the training configuration, and the evaluation metrics at the time of registration. Serve as the handoff point between the training pipeline and the inference system.


### API Specifications

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/jobs` | Submit images + config, start training job |
| `GET` | `/jobs/{job_id}` | Poll job status and result |
| `GET` | `/jobs` | List all jobs (paginated) |
| `DELETE` | `/jobs/{job_id}` | Cancel a running job |
| `GET` | `/jobs/{job_id}/progress` | SSE stream of training steps |
| `POST` | `/evaluate` | Evaluate an existing model |
| `POST` | `/ab-test` | Compare two models |
| `GET` | `/health` | Service health + GPU availability |
| `GET` | `/metrics` | Prometheus-format operational metrics |


### Data Models and Schemas

**TrainingRequest** — parameters supplied by the user at job submission
 
| Field | Type | Description |
|-------|------|-------------|
| `trigger_word` | string | The token representing the trained concept |
| `base_model_id` | string | Identifier of the base diffusion model |
| `lora_rank` | integer | Rank of the low-rank adapter matrices |
| `learning_rate` | float | Optimiser learning rate |
| `num_epochs` | integer | Number of full passes over the training set |
| `target_size` | integer | Image resolution used during training (pixels) |
 
**ImageRecord** — represents a single image after processing
 
| Field | Type | Description |
|-------|------|-------------|
| `path` | string | Location of the processed image file |
| `caption` | string | Text annotation associated with the image |
| `split` | string | Dataset partition: `train` / `val` / `test` |
| `is_augmented` | boolean | Whether this is a synthetically generated variant |
 
**JobRecord** — tracks the state of a training job
 
| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique job identifier |
| `status` | enum | `pending` / `processing` / `training` / `evaluating` / `completed` / `failed` |
| `progress` | object | Current stage and completion percentage |
| `result` | object | Populated on successful completion |
| `error` | string | Populated on failure, null otherwise |
 
**TrainingStep** — single progress event emitted during training
 
| Field | Type | Description |
|-------|------|-------------|
| `step` | integer | Global step count |
| `epoch` | integer | Current epoch |
| `loss` | float | Training loss at this step |
| `lr` | float | Current learning rate |
| `elapsed_seconds` | float | Wall-clock time since training started |
 
**EvalMetrics** — output of the evaluation component
 
| Field | Type | Description |
|-------|------|-------------|
| `clip_score` | float | Semantic alignment between generated images and prompts |
| `fid_score` | float / null | Distributional distance from real training images; null if unavailable |
| `passes_threshold` | boolean | Whether the model meets minimum quality requirements |
 
**TrainingResult** — final output of a completed training job
 
| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Identifier of the originating job |
| `model_path` | string | Location of the saved model artifact |
| `final_loss` | float | Training loss at the last step |
| `summary` | object | Total steps, duration, and training configuration |
| `success` | boolean | Whether training completed without error |
| `error` | string | Error message if success is false, null otherwise |


### Error Handling Strategies

**Input validation**
All inputs are validated at the API boundary before any processing begins. Invalid file formats, out-of-range parameter values, and malformed requests are rejected immediately with a descriptive error response. Jobs are never created for requests that fail validation.

**Image-level rejection**
Individual images that fail quality checks are silently excluded from the dataset rather than failing the entire job. The job proceeds as long as the remaining valid image count meets the minimum threshold. If the valid set falls below the minimum, the job is terminated with a clear explanation of how many images were rejected and why.

**Job-level failures**
Any failure during processing, training, or evaluation transitions the job to a failed state and records the cause. The failure is propagated to the client via the SSE stream as a terminal event. Failures in one job must not affect other running or queued jobs.

**Interrupted training**
Training jobs that are interrupted mid-run must be resumable from the most recent checkpoint. The system should detect the presence of a prior checkpoint on startup and continue from that point rather than restarting from scratch.

**Resource exhaustion**
When GPU memory is insufficient to run a job with the requested configuration, the job should fail with an actionable error message suggesting configuration adjustments (e.g. reducing batch size or image resolution) rather than crashing the service.

## Quick Start

### Local (CPU / stub mode — no GPU required for testing)

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### Docker (GPU)

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

---

## API Reference

### Submit a training job

```bash
curl -X POST http://localhost:8000/jobs \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  ... \
  -F 'config={"trigger_word":"sks","lora_rank":16,"num_epochs":100}'
```

Response:
```json
{"job_id": "abc123", "status": "pending"}
```

### Poll job status

```bash
curl http://localhost:8000/jobs/abc123
```

### Stream live training progress (SSE)

```bash
curl -N http://localhost:8000/jobs/abc123/progress
```

Each event:
```
event: step
data: {"step": 200, "epoch": 2, "loss": 0.142, "lr": 9.8e-05, "elapsed_seconds": 45.2}

event: done
data: {"status": "completed", "result": {...}}
```

### List all jobs

```bash
curl http://localhost:8000/jobs
```

### Cancel a job

```bash
curl -X DELETE http://localhost:8000/jobs/abc123
```

### Evaluate a model

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model_path": "outputs/abc123/models/abc123"}'
```

### A/B test two models

```bash
curl -X POST http://localhost:8000/ab-test \
  -H "Content-Type: application/json" \
  -d '{
    "model_a_path": "outputs/job1/models/job1",
    "model_b_path": "outputs/job2/models/job2",
    "model_a_id": "v1",
    "model_b_id": "v2"
  }'
```

### Health & metrics

```bash
curl http://localhost:8000/health
curl http://localhost:8000/metrics   # Prometheus text format
```

---

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trigger_word` | `sks` | Token used to identify the trained concept |
| `base_model_id` | `runwayml/stable-diffusion-v1-5` | HuggingFace model ID |
| `lora_rank` | `16` | LoRA rank (higher = more parameters) |
| `lora_alpha` | `32` | LoRA scaling factor |
| `learning_rate` | `1e-4` | AdamW learning rate |
| `num_epochs` | `100` | Training epochs |
| `batch_size` | `1` | Per-GPU batch size |
| `target_size` | `512` | Image resolution for training |

---

## Data Processing Pipeline

1. **Validation** — format check, minimum size (256×256), file integrity, blur detection (Laplacian variance)
2. **Deduplication** — MD5 hash comparison across the entire upload batch
3. **Preprocessing** — center-crop to square, resize to `target_size`, RGB conversion
4. **Captioning** — filename-derived context + trigger word. Swap `CaptionGenerator.generate()` for a BLIP/CogVLM call to get richer captions.
5. **Augmentation** — horizontal flip, ±10° rotation, brightness jitter (configurable multiplier)
6. **Splitting** — deterministic 85/10/5 train/val/test split (seed-controlled)
7. **Caption files** — writes `.txt` sidecars alongside each image (diffusers convention)

---

## Evaluation

| Metric | Tool | Threshold (default) |
|--------|------|---------------------|
| CLIP score | openai/clip ViT-B/32 | ≥ 0.20 |
| FID | clean-fid / torch-fidelity | ≤ 200 |
| Inference speed | timed via `time.perf_counter` | logged |

Install optional eval deps for full metrics:

```bash
pip install openai-clip clean-fid
```

Without them, CLIP falls back to a calibrated stub and FID is skipped.

---

## Running Tests

```bash
pytest tests/ -v
```

Run the data processing benchmark:

```bash
python tests/test_pipeline.py
```

---

## GPU Resource Management

- The trainer detects available GPUs via `torch.cuda.device_count()`
- Mixed precision (`fp16`) is used by default to halve VRAM usage
- Gradient checkpointing is enabled by default for large models
- `xformers` memory-efficient attention is opt-in (`enable_xformers: true`)
- The checkpoint manager keeps at most 3 checkpoints and prunes older ones automatically

For multi-GPU training across 2–4 GPUs, replace `BackgroundTasks` with a Celery worker pool and pass `--num_processes N` to `accelerate launch`.

---

## Production Considerations

- **Secrets**: mount a `.env` file or use Kubernetes Secrets for HuggingFace tokens
- **Persistence**: mount a shared volume (NFS / S3-fuse) for `outputs/` across replicas
- **Observability**: the `/metrics` endpoint is Prometheus-compatible; add Grafana for dashboards
- **Scaling**: replace the in-process `BackgroundTasks` job store with Redis + Celery for horizontal scaling
- **Model registry**: integrate MLflow or W&B for versioned artifact tracking