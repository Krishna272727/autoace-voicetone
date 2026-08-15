# Single container, models baked in at build time.
#
# Nothing is downloaded at runtime: a cold start that has to fetch Whisper and
# a SER checkpoint takes tens of seconds and will time out behind a load
# balancer (BUILD_SPEC.md 11.7).

FROM python:3.11-slim

# ffmpeg is not optional -- every decode path goes through it.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.hf_cache \
    VOICETONE_MODELS=/app/models \
    OMP_NUM_THREADS=1 \
    TORCH_THREADS=1

# CPU-only torch: roughly a tenth the size of the default CUDA build and the
# only thing that fits a 4 GB instance alongside the checkpoints.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchaudio \
         --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Order matters. The model download below is a ~20 minute layer, and Docker
# invalidates every layer after the first changed COPY. `app/` is the code that
# actually changes between deploys, so it is copied LAST -- otherwise editing a
# template re-downloads 2 GB of weights. Only `voicetone/` and `scripts/` are
# needed to run the downloader.
COPY voicetone/ ./voicetone/
COPY scripts/ ./scripts/
COPY config/ ./config/
COPY cli.py ./

# Bake every weight into the image, Whisper included.
#
# This previously ran --skip-asr, which contradicted the header above: the
# lexical-valence path would have fetched 464 MB from Hugging Face on the first
# ambiguous call, mid-request, behind a load balancer. That is exactly the cold
# start BUILD_SPEC 11.7 rules out. It costs ~464 MB of image to remove it.
RUN python scripts/download_models.py \
    && python scripts/download_models.py --prune

# Prove the claim in the header rather than assuming it. With the hub forced
# offline, loading every checkpoint either succeeds from the baked cache or
# fails the build here -- it cannot quietly fall back to the network.
#
# This exists because it did quietly fall back. A per-repo prune removed the
# only weights on `main` for the sentiment checkpoint, and transformers
# re-resolved it from an unmerged pull request on huggingface.co at every cold
# start: ~10 s of warm-up, and outbound requests LICENCES.md says do not happen.
# Nothing caught it, because the fallback logs as success.
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python scripts/download_models.py

# Runtime is offline too. The cache is complete by construction above, so this
# is belt-and-braces: any future gap fails loudly at startup instead of turning
# into a silent dependency on a third party.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Application code last: a change here rebuilds only the layers below.
COPY app/ ./app/

# Non-root: the container handles uploaded customer audio.
# UID 1000 because Hugging Face Spaces runs Docker containers as that user and
# anything owned by another UID is unwritable there.
RUN useradd --create-home --uid 1000 autoace \
    && chown -R autoace:autoace /app
USER autoace
ENV HOME=/home/autoace

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# One worker: the models are process-resident and a second copy would double
# the memory. Batch parallelism happens inside the process across files.
#
# Shell form, not exec form, so ${PORT} expands. Cloud Run and most container
# platforms inject the port to listen on rather than letting the image choose;
# an exec-form CMD would take the string "${PORT}" literally and never bind.
# Defaults to 8000 so `docker run -p 8000:8000` still works unchanged.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} \
    --workers 1 --timeout-keep-alive 75
