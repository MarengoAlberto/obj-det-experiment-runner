FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Add OS-level dependencies only when you actually need them.
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Keep this before copying the rest of the code so Docker can cache dependencies.
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

#ENTRYPOINT ["bash", "/app/run_training.sh"]
CMD ["bash", "-lc", "sleep infinity"]