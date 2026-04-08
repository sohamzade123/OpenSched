FROM python:3.11-slim

WORKDIR /app

# Ensure Python output is sent straight to the container logs
ENV PYTHONUNBUFFERED=1

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app/ ./app/
COPY tests/ ./tests/
COPY inference.py openenv.yaml ./

EXPOSE 7860

# Run uvicorn as the main foreground process (PID 1, exec form)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
