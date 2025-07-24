FROM python:3.9-slim

# التبعيات الأساسية فقط
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# تثبيت Ollama
RUN wget https://ollama.com/download/ollama-linux-amd64 -O /usr/bin/ollama \
    && chmod +x /usr/bin/ollama

WORKDIR /app
COPY . .

# تثبيت المتطلبات
RUN pip install --no-cache-dir -r requirements.txt

ENV OLLAMA_HOST=0.0.0.0:11434
ENV PYTHONUNBUFFERED=1

CMD ["/app/setup.sh"]