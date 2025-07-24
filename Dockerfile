FROM python:3.9-slim

# تثبيت التبعيات النظامية الأساسية
RUN apt-get update && apt-get install -y \
    wget \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# تنزيل وتثبيت Ollama
RUN wget https://ollama.com/download/ollama-linux-amd64 -O /usr/bin/ollama \
    && chmod +x /usr/bin/ollama

# نسخ ملفات التطبيق
WORKDIR /app
COPY . .

# تثبيت متطلبات Python (بما فيها sentencepiece)
RUN pip install --no-cache-dir -r requirements.txt

# تثبيت المكتبات الإضافية المطلوبة
RUN pip install --no-cache-dir \
    sentencepiece \
    protobuf
RUN python -c "import sentencepiece; print('SentencePiece installed successfully')"
# تعيين متغيرات البيئة
ENV OLLAMA_HOST=0.0.0.0:11434
ENV PYTHONUNBUFFERED=1

# جعل ملف setup.sh قابلاً للتنفيذ
RUN chmod +x /app/setup.sh

# تشغيل البرنامج النصي للإعداد عند بدء التشغيل
CMD ["/app/setup.sh"]