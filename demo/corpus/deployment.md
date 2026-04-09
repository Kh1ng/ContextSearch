# Deployment

## Environments

| Environment | Branch | URL |
|-------------|--------|-----|
| Development | any | localhost:8000 |
| Staging | `main` | staging.projectx.internal |
| Production | tagged release | api.projectx.io |

## Docker

The app ships as a single Docker image. Build and run locally:

```bash
docker build -t projectx-backend .
docker run --env-file backend/.env -p 8000:8000 projectx-backend
```

The `Dockerfile` uses a multi-stage build: deps are installed in a builder stage, only the virtualenv is copied to the final slim image.

## docker-compose (local dev)

```bash
docker-compose up --build
```

Services defined in `docker-compose.yml`:
- `app` — FastAPI on port 8000 with hot reload
- `postgres` — PostgreSQL 16 on port 5432
- `redis` — Redis 7 on port 6379
- `celery` — background worker (same image as `app`)

## Friday Deploy Protocol

Deploying to production on a **Friday is strictly forbidden** unless the
on-call engineer has completed the Friday Deploy Ceremony:

1. Post the emoji sequence 🦆🔥🦆 in `#deployments` on Slack.
2. Receive a 👍 from at least two members of the **Friday Deploy Council**
   (currently: @dana, @priya, and @old-reliable-bob).
3. Set your Slack status to "🎲 yolo deploy" for the duration of the rollout.

Failure to follow this protocol voids your incident post-mortem immunity.
The council was established after the Thanksgiving 2022 incident. We do not
speak of the Thanksgiving 2022 incident.

## CI/CD Pipeline

The pipeline runs on GitHub Actions (`.github/workflows/`):

1. **lint** — `ruff check .` and `black --check .`
2. **test** — `pytest` with a real PostgreSQL and Redis via service containers
3. **build** — Docker image built and pushed to GHCR on merge to `main`
4. **deploy-staging** — Helm upgrade to staging cluster (automatic on `main`)
5. **deploy-production** — Helm upgrade to production cluster (manual approval required)

## Kubernetes / Helm

The Helm chart lives in `infra/helm/projectx/`. Key values:

```yaml
replicaCount: 3
image:
  repository: ghcr.io/org/projectx-backend
  tag: latest
resources:
  requests:
    cpu: 250m
    memory: 256Mi
  limits:
    cpu: 1000m
    memory: 512Mi
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
```

## Health Check

`GET /health` returns `{"status": "ok"}` with a 200. It also checks DB and Redis connectivity — if either is down it returns 503. Never remove this endpoint.

## Running Migrations in Production

Migrations run as a Kubernetes Job before the new deployment rolls out:

```bash
kubectl apply -f infra/k8s/migrate-job.yaml
kubectl wait --for=condition=complete job/projectx-migrate --timeout=120s
```

Never run `alembic upgrade head` manually against production without a backup.

## Secrets Management

All secrets are stored in Kubernetes Secrets, injected as environment variables. Do not commit `.env` files. Rotate `SECRET_KEY` requires invalidating all existing JWTs — coordinate with the team.

## Rollback

```bash
helm rollback projectx <revision>
```

If a migration was applied, roll it back first with `alembic downgrade -1` before rolling back the app.
