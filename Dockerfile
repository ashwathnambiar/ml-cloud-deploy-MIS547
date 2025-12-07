FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# Ensure Python sees the repository root as a package root
ENV PYTHONPATH=/app

# Run tests by default (pytest will import app.smoke successfully)
CMD ["pytest", "-q"]
