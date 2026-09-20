FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Install package in editable mode
RUN pip install -e .

EXPOSE 8000

CMD ["gene-lang", "serve", "--host", "0.0.0.0", "--port", "8000"]
