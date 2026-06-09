# PhytoLabs FastAPI inference server
FROM python:3.11-slim

WORKDIR /app

# OpenCV runtime deps (headless wheels still need libGL on slim images).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir -e ".[api]"

# Bootstrap artifacts at image build so Railway deploy works without a manual upload.
# Replace later: mount a Railway volume at /app/artifacts with Colab-trained gmm.joblib + logreg.joblib.
RUN mkdir -p artifacts /tmp/synth-data /tmp/synth-artifacts && \
    python -m phytolabs.cli make-synthetic --data-dir /tmp/synth-data --n-per-class 12 && \
    python -m phytolabs.cli train-gmm --data-dir /tmp/synth-data --gmm artifacts/gmm.joblib && \
    python -m phytolabs.cli build-features --data-dir /tmp/synth-data --artifacts /tmp/synth-artifacts && \
    python -m phytolabs.cli train-logreg --artifacts /tmp/synth-artifacts --logreg artifacts/logreg.joblib

ENV PHYTOLABS_GMM=/app/artifacts/gmm.joblib
ENV PHYTOLABS_LOGREG=/app/artifacts/logreg.joblib
ENV PHYTOLABS_CORS_ORIGINS=*

EXPOSE 8000

CMD ["python", "-m", "phytolabs.api.main"]
