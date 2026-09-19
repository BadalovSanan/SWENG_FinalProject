FROM python:3.13.15-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies before copying source so code edits reuse this layer.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY . .

ENTRYPOINT ["python", "-m", "researcher"]
CMD ["--help"]
