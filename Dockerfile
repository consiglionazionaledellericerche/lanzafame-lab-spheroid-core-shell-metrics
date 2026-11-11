FROM python:3.11-slim

# libgl1 / libglib2.0-0 are required by opencv-python-headless at import time
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run Docker containers as uid 1000, not root — create
# that user and own the app dir, otherwise Spaces deploys can hit permission
# errors. Harmless on Render/Railway too (they don't care either way).
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the whole folder so the optional static/ (lab logo) is included even
# when it is absent from the repo (an explicit COPY static/ would then fail).
COPY --chown=user . ./

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PORT=7860
EXPOSE 7860

# Shell form so $PORT is honoured on platforms that inject their own
# (Render, Railway, Fly.io); defaults to 7860, which is what Hugging Face
# Spaces' Docker SDK expects.
CMD gunicorn -b 0.0.0.0:$PORT -w 2 --timeout 120 app:app
