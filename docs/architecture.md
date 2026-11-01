# Architecture

TruthLens has two halves that meet at a versioned model registry. Training happens on free GPUs in notebooks and scripts. Serving happens on CPU behind an asynchronous API.

## System diagram

```mermaid
flowchart TB
  subgraph training["Training (Colab or Kaggle GPU)"]
    data["Datasets and manifest CSVs"]
    train["Hydra configs and PyTorch training"]
    track["Weights & Biases runs"]
    export["Evaluation, calibration, ONNX export"]
    data --> train --> track
    train --> export
  end

  subgraph clients["Clients"]
    web["React web app"]
    ext["Chrome extension (MV3)"]
    public["Public API users"]
  end

  subgraph edge["API layer"]
    api["FastAPI: auth, validation, rate limits, WebSocket"]
  end

  subgraph state["State"]
    pg[("PostgreSQL")]
    minio[("MinIO: uploads, artefacts, reports")]
    redis[("Redis: broker, pub/sub, rate limits")]
  end

  subgraph workers["Celery workers (CPU)"]
    fast["fast queue: image and text"]
    audio["audio queue"]
    reports["reports queue: WeasyPrint PDF"]
    beat["Celery beat: 24-hour media purge"]
  end

  registry["models.yaml: ONNX and PyTorch weights by version"]

  web -->|"JWT, HTTPS and WebSocket"| api
  ext -->|"API key, HTTPS with polling"| api
  public -->|"API key, HTTPS"| api
  api --> pg
  api --> minio
  api -->|"enqueue job"| redis
  redis --> fast
  redis --> audio
  redis --> reports
  beat --> redis
  fast --> pg
  fast --> minio
  audio --> pg
  audio --> minio
  reports --> pg
  reports --> minio
  fast -->|"progress events"| redis
  audio -->|"progress events"| redis
  redis -->|"pub/sub to WebSocket"| api
  export -->|"publish versioned models"| registry
  registry --> fast
  registry --> audio
```

## Request flow

1. The client posts a file, text or URL. The API authenticates it and checks the rate limit.
2. For direct uploads, the API checks magic numbers and size limits, strips EXIF and runs the virus-scan hook. A URL job runs those same checks in the worker after the download.
3. The media goes to MinIO and an `analyses` row is created with status `queued`. The API returns `202` with the analysis id.
4. A Celery worker runs ONNX inference, then builds the explanation with the PyTorch weights.
5. The worker writes `analysis_results`, stores artefacts in MinIO and publishes progress to Redis.
6. The API relays events to the browser over WebSocket. The extension polls `GET /analyses/{id}` instead.
7. Celery beat deletes unsaved media, artefacts and stored source text 24 hours after upload.

## Monorepo structure

```text
TruthLens/
├── ml/
│   ├── common/             # Seeds, manifests, splits, samplers, metrics, data cards
│   │   └── tests/          # Includes the no-file-in-two-splits tests
│   ├── image/              # Models, training, evaluation, export, explain.py
│   │   └── configs/
│   ├── audio/              # Features, models, training, evaluation, export, saliency
│   │   └── configs/
│   ├── text/               # Statistical features, transformers, ensemble, robustness
│   │   └── configs/
│   └── eval/               # robustness.py, results notebook
│       └── results/        # Results CSVs consumed by the paper
├── backend/
│   ├── app/
│   │   ├── api/            # Routers and WebSocket handlers
│   │   ├── core/           # Settings, auth, rate limits, logging
│   │   ├── db/             # SQLAlchemy models and Alembic migrations
│   │   ├── inference/      # Model registry, ONNX sessions, per-modality pipelines
│   │   ├── reports/        # Jinja2 templates, QR codes, HMAC signing
│   │   ├── storage/        # MinIO client
│   │   ├── validation/     # Magic numbers, EXIF, URL fetching, virus-scan hook
│   │   └── worker/         # Celery app, tasks, purge schedule
│   ├── models.yaml
│   └── tests/
├── frontend/
│   └── src/
│       ├── api/            # Typed client and React Query hooks
│       ├── components/     # Gauge, heatmap, waveform, sentence highlights
│       ├── pages/          # Landing, Analyze, Result, History, API keys, Admin, Verify
│       └── test/
├── extension/
│   ├── src/
│   │   ├── background/     # Service worker and context menus
│   │   ├── overlay/        # Shadow DOM result overlay
│   │   ├── options/
│   │   └── popup/
│   └── scripts/            # Packages the .zip
├── infra/
│   ├── docker/             # Dockerfiles for api, worker, web
│   └── docker-compose.yml
├── paper/
│   ├── figures/
│   └── tables/
├── docs/
└── .github/workflows/
```

The master prompt lists `ml/image`, `ml/audio`, `ml/text` and `ml/common`. `ml/eval` is added because prompts 2.10 and 2.11 require `ml/eval/robustness.py` and `ml/eval/results/`.
