# Docker +5 optional extension.
# Packages the trained ETA model and its FastAPI service so a reviewer can run
# the prototype without installing Python or any dependency by hand.
#
# Build and run:
#   docker build -t taxi-eta .
#   docker run --rm -p 8000:8000 taxi-eta
#
# The raw Kaggle dataset is never copied in (see .dockerignore); the image
# carries only the fitted artifact, which holds coefficients, not trip rows.

FROM python:3.12-slim

# Keeps .pyc files out of the image and makes log output appear immediately
# rather than sitting in a buffer when the container is stopped.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies are copied and installed before the source so that editing the
# code does not invalidate the cached install layer on the next build.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY src/ ./src/
COPY tests/ ./tests/
COPY artifacts/ ./artifacts/

# The service writes its monitoring log to logs/. Create it up front and hand
# it to the unprivileged user, so the container does not need to run as root.
RUN mkdir -p logs \
    && useradd --create-home --shell /usr/sbin/nologin appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# 0.0.0.0 rather than 127.0.0.1: bound to loopback the service would only be
# reachable from inside the container, and -p 8000:8000 would appear to fail.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
