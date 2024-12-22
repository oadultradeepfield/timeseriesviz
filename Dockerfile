# Use a slim base image
FROM python:3.11-slim

# Set environment variables to reduce Python overhead
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy only requirements first (to cache dependencies during rebuilds)
COPY requirements.txt .

# Install dependencies with no cache and clean up apt lists
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && apt-get install libgomp1 \
    && pip3 install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application
COPY . .

# Expose the port for the app
EXPOSE 8080

# Run the application
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
