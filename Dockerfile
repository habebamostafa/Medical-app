FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir sentence-transformers

# Install Ollama
RUN wget https://ollama.com/download/ollama-linux-amd64 -O /usr/bin/ollama \
    && chmod +x /usr/bin/ollama

# Set up application
WORKDIR /app
COPY . .

# Install Python requirements
RUN pip install --no-cache-dir -r requirements.txt

# Environment variables
ENV OLLAMA_HOST=0.0.0.0:11434
ENV PYTHONUNBUFFERED=1

# Create startup script
RUN echo '#!/bin/bash\n\
# Start Ollama in background\n\
ollama serve > /var/log/ollama.log 2>&1 &\n\
\n\
# Wait for server to start\n\
while ! curl -s http://localhost:11434 >/dev/null; do\n\
  echo "Waiting for Ollama to start..."\n\
  sleep 1\n\
done\n\
\n\
# Pull the model\n\
ollama pull deepseek-r1:1.5b\n\
\n\
# Start Streamlit\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/setup.sh && \
    chmod +x /app/setup.sh

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=30s \
  CMD curl -f http://localhost:11434 || exit 1

# Run the application
CMD ["/app/setup.sh"]