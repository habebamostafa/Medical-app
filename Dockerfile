FROM python:3.9-slim

# تثبيت التبعيات النظامية
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# تنزيل وتثبيت Ollama
RUN wget https://ollama.com/download/ollama-linux-amd64 -O /usr/bin/ollama \
    && chmod +x /usr/bin/ollama

# نسخ ملفات التطبيق
WORKDIR /app
COPY . .

# تثبيت متطلبات Python
RUN pip install --no-cache-dir -r requirements.txt

# تعيين متغيرات البيئة
ENV OLLAMA_HOST=0.0.0.0:11434

# تشغيل Ollama وتنزيل النموذج عند بدء الحاوية
COPY setup.sh /app/setup.sh
RUN chmod +x /app/setup.sh

# تشغيل البرنامج النصي للإعداد عند بدء التشغيل
CMD ["/app/setup.sh"]

RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    && pip install --no-cache-dir sentencepiece