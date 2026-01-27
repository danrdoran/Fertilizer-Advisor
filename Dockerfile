# Dockerfile
# Uses a non-root WORKDIR (/app) which Streamlit recommends for container runs.
# Installs libgomp1 (OpenMP runtime) needed by LightGBM / XGBoost / CatBoost.

FROM python:3.12-slim

# Prevents Python from writing .pyc files and keeps logs unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies:
# - libgomp1: provides /usr/lib/**/libgomp.so.1 (OpenMP runtime)
# - build-essential + cmake: helps if any wheels need to compile (platform/arch edge cases)
# - curl: used for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        build-essential \
        cmake \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of the repo
COPY . /app

# Streamlit default port
EXPOSE 8501

# Streamlit health endpoint (per Streamlit docs)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "scripts/fertilizer_advisor_app.py", "--server.port=8501", "--server.address=0.0.0.0"]