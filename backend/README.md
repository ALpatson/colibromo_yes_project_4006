# backend/: Django + DRF API and Celery jobs

> Placeholder (Step 0). Implemented in **Step 5**.

Exposes scenes, captures, reconstruction jobs and variants over a REST API (`/api/v1/`), hands out presigned S3 URLs for uploads/downloads, and runs the pipeline in a Celery worker on the GPU host.

Planned layout (PRD §6):

```
backend/
├── manage.py
├── config/      # settings, urls, celery app
├── scenes/      # Scene, SceneVariant, Capture, ReconstructionJob models + API
└── tests/
```

Uses the PostgreSQL, Redis and MinIO services from the root `docker-compose.yml`.
