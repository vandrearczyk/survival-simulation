FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing bytecode or buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml README.md /app/
COPY src/ /app/src/

RUN pip install --no-cache-dir .

ENTRYPOINT ["simulate-survival"]