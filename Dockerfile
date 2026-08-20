FROM python:3.11-slim

WORKDIR /app

# Copy dependency manifest
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

EXPOSE 8000

ENV HOST=0.0.0.0
ENV PORT=8000
ENV ENV=production

CMD ["python", "server.py"]
