# Deployment

## What is running: Google Cloud Run

```bash
gcloud run deploy autoace --source . --region us-central1 \
  --memory 8Gi --cpu 8 --no-cpu-throttling --cpu-boost \
  --min-instances 1 --max-instances 1 \
  --timeout 3600 --concurrency 20 --allow-unauthenticated \
  --set-env-vars "DASHBOARD_USER=...,BATCH_WORKERS=2,OMP_NUM_THREADS=4,TORCH_THREADS=4,COOKIE_SECURE=1" \
  --set-secrets "DASHBOARD_PASS=autoace-pass:latest,SESSION_SECRET=autoace-session:latest"
```

Four of those flags are load-bearing and were each found by something breaking.

**`--no-cpu-throttling`.** Cloud Run allocates CPU only while a request is in
flight. Analysis runs in a background thread pool after the upload response is
sent, so without this the job stalls the moment the response returns: observed
sitting at 28% indefinitely, creeping forward only on the CPU slices granted by
status polls.

**`--max-instances 1`.** The job store is an in-process dict. A second instance
would 404 status polls for jobs it never saw. This is the same property that
rules out serverless functions entirely; Cloud Run just lets you pin it.

**`--min-instances 1`.** With scale-to-zero there is no process to run the
background pool between requests.

**`--cpu-boost`.** Extra CPU during startup, which is when the models load.

Credentials live in Secret Manager rather than env vars, so they are not
visible in the console or in deploy logs.

### Cost

Always-allocated CPU is billed continuously, so the instance size *is* the
running cost:

| | approx / month |
|---|---|
| 2 vCPU + 8 GiB | $176 |
| 4 vCPU + 8 GiB | $300 |
| 8 vCPU + 8 GiB | $550 |

Measured warm throughput: **RTF 0.47 at 8 vCPU, 1.88 at 2 vCPU**. Scale up for
an evaluation window, then `--min-instances 0` to drop the cost to near zero.

### Two things that cost hours, recorded so they do not again

**Cold start dominates the first request.** The same 31-second call took **183 s
cold and 19.6 s warm**, because checkpoints were `lru_cache`d and loaded lazily
on first use. Every deploy makes a new revision, so the first person to open the
link always paid it and concluded the system was slow. `app/main.py` now warms
the models on a background thread at startup.

**`/healthz` never reaches the container.** Google's frontend intercepts that
exact path and answers its own 404 — recognisable because the response carries
no `server: Google Frontend` header, unlike everything that does reach the app.
The health endpoint is served at `/health` as well.

## Why not Vercel

Vercel caps a serverless function at **250 MB unzipped**. Measured on this
machine, `torch` alone is 533 MB and the five production checkpoints are 3.2 GB
of weights. Compression does not close a 13× gap.

Three further blockers, none of which are about size:

- **Functions are stateless.** The job store is an in-process dict plus a
  `ThreadPoolExecutor`. A `/jobs/{id}/status` poll would land on an invocation
  that never saw the job.
- **Execution is request-scoped.** At the measured 0.43× real time, a 30-minute
  recording is 13 minutes of compute, and a batch is far more.
- **No persistent disk** for the 60-minute playback retention window.

The same reasoning rules out every serverless-function host. This needs a
long-lived container, which is what `BUILD_SPEC.md` §11.7 specifies.

## Hugging Face Spaces — needs PRO, verified against the API

Spaces was the obvious free candidate and it no longer is. Creating a Docker
Space returns **402 Payment Required**:

> Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on
> free cpu-basic requires a PRO subscription.

Tested both public and private; the gate is the Docker SDK, not visibility.
Static Spaces are free but cannot run Python, so they are no use here. PRO is
$9/month and the Space then gets 2 vCPU / 16 GB RAM, which is ample.

If you subscribe, everything below works unchanged.

```bash
pip install -U "huggingface_hub[cli]"
hf auth login

# Create a Docker Space (private keeps evaluation audio out of public view)
hf repo create autoace-voicetone --repo-type space --space_sdk docker --private

git clone https://huggingface.co/spaces/<your-user>/autoace-voicetone /tmp/space
cd /tmp/space

# Everything the image needs. .dockerignore keeps the venv and caches out.
rsync -a --exclude .git --filter=':- /path/to/repo/.dockerignore' \
      /path/to/repo/ .

# The Space is configured by YAML frontmatter in its README
cp deploy/space-README.md README.md

git add -A && git commit -m "AutoAce voice tone dashboard" && git push
```

Then set three secrets under **Settings → Variables and secrets**:

| Secret | Why |
|---|---|
| `DASHBOARD_USER` | otherwise the shipped default is active and the UI says so |
| `DASHBOARD_PASS` | same |
| `SESSION_SECRET` | without it a restart invalidates every signed cookie |

Optional: `COOKIE_SECURE=1` (Spaces serve over HTTPS), `AUDIO_RETENTION_MIN=0`
to delete uploads the moment a batch finishes and disable playback.

The first build takes roughly 15–25 minutes; it installs CPU torch and bakes
every checkpoint into the image so there is no runtime download.

### Two things to disclose, not hide

**Audio transits a third party.** `BUILD_SPEC.md` §11.7 flags this directly:
Spaces is the fastest path to a live URL, "**but** evaluation audio would
transit a third-party public service, which sits awkwardly against the brief's
data-handling clause. If you use it, disclose it explicitly." The architecture
is still local-only inference with no paid API, but the host is not your
infrastructure. Say so in the memo.

**Free Spaces pause when idle.** §11.7 says to avoid tiers that sleep, because
the brief requires availability through the evaluation period. A paused Space
wakes on the next visit, but the first request after a pause is slow. If the
evaluation window matters more than the cost, upgrade the Space hardware for
those days and downgrade after.

## Genuinely free, if you will do the setup: Oracle Cloud Always Free

The only remaining $0 option that can hold a 3 GB image. The Always Free tier
gives 4 ARM cores and 24 GB RAM on an Ampere A1 instance, indefinitely, which is
six times the 4 GB target. A card is required for identity verification but is
not charged.

The cost is effort: it is a bare VM, not a platform. Create the instance, open
port 8000 in the security list, install Docker, then

```bash
docker build -t autoace .
docker run -d --restart unless-stopped -p 8000:8000 \
  -e DASHBOARD_USER=... -e DASHBOARD_PASS=... \
  -e SESSION_SECRET="$(openssl rand -hex 32)" autoace
```

Note the image is built for `linux/amd64` by default; on Ampere build natively
on the instance rather than cross-building, or pass `--platform linux/arm64`.
Nothing in the stack is x86-specific.

## No deployment at all

Worth stating as a real option rather than a failure. BUILD_SPEC 15 puts the
dashboard at 10% and asks for it "deployed and reachable", so this does cost
marks — but a reviewer with Docker can be running it in one command:

```bash
docker build -t autoace . && docker run -p 8000:8000 \
  -e DASHBOARD_USER=you -e DASHBOARD_PASS='pick-something' autoace
```

That is most of the demonstrable value, and it keeps every byte of evaluation
audio on the reviewer's own machine, which is the strongest version of the
data-handling story.

## Paid, if the privacy story matters more

`BUILD_SPEC.md` §11.7 names Fly.io, Render and Railway at ~4 GB RAM and budgets
$10–25/month. Fly is the closest fit: Docker-native, builds remotely so no local
Docker daemon is needed, and it does not sleep with `min_machines_running = 1`.

```bash
fly launch --no-deploy --name autoace-voicetone
fly scale memory 4096
fly secrets set DASHBOARD_USER=... DASHBOARD_PASS=... SESSION_SECRET="$(openssl rand -hex 32)"
fly deploy --remote-only
```

## Image size

`scripts/download_models.py --prune` removes `pytorch_model.bin` weights
superseded by `model.safetensors`. Several of these repos publish only `.bin` on
`main`, so transformers follows the safetensors-conversion pull request and the
cache ends up holding both formats in different revisions. Measured saving:

```
  478 MB  cardiffnlp/twitter-roberta-base-sentiment-latest
  386 MB  microsoft/wavlm-base-plus-sv
  361 MB  superb/wav2vec2-base-superb-er
 1224 MB  total
```

The Dockerfile runs it after the download step. It assumes transformers keeps
choosing the safetensors revision; if a repo removes it, rebuild the image.
