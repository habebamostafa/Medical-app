#!/bin/bash

# Redirect all output to a log file
exec > >(tee -a /var/log/app-startup.log) 2>&1

# Function to check Ollama health
check_ollama_health() {
    local max_retries=10
    local retry_interval=3
    local retries=0
    
    echo "Checking Ollama server health..."
    until curl -s http://localhost:11434 >/dev/null; do
        retries=$((retries+1))
        if [ $retries -ge $max_retries ]; then
            echo "Ollama server failed to start after $max_retries attempts"
            exit 1
        fi
        echo "Waiting for Ollama to start... (Attempt $retries/$max_retries)"
        sleep $retry_interval
    done
    echo "Ollama server is ready"
}

# Start Ollama in background with logging
echo "Starting Ollama server..."
ollama serve >> /var/log/ollama.log 2>&1 &
OLLAMA_PID=$!

# Check if Ollama started successfully
check_ollama_health

# Pull the model with retry logic
echo "Pulling deepseek-r1:1.5b model..."
MAX_RETRIES=3
RETRY_DELAY=5
for i in $(seq 1 $MAX_RETRIES); do
    if ollama pull deepseek-r1:1.5b; then
        echo "Model pulled successfully"
        break
    else
        echo "Model pull failed (attempt $i/$MAX_RETRIES)"
        if [ $i -eq $MAX_RETRIES ]; then
            echo "Failed to pull model after $MAX_RETRIES attempts"
            exit 1
        fi
        sleep $RETRY_DELAY
    fi
done

# Start Streamlit application
echo "Starting Streamlit application..."
exec streamlit run app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false