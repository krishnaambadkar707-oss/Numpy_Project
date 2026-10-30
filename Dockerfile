FROM python:3.11-slim

WORKDIR /app

# Copy dependency manifest
COPY requirements.txt .

# Install ffmpeg system package for audio decoding and conversion
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000
ENV ENV=production

CMD ["python", "server.py"]
