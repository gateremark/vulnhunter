# VulnHunter OpenEnv Environment
FROM python:3.10-slim

# Install security tools and dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    sqlmap \
    nmap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY vulnhunter/ /app/vulnhunter/
COPY vulnhunter/vulnerable_app/ /app/vulnerable_app/

# Create files directory for path traversal demo
RUN mkdir -p /app/files && echo "Welcome to VulnHunter" > /app/files/readme.txt

# Expose OpenEnv port
EXPOSE 8000

# Start the environment server
CMD ["uvicorn", "vulnhunter.env_server.server:app", "--host", "0.0.0.0", "--port", "8000"]
