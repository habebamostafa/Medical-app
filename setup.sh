FROM python:3.9-slim

# 1. تثبيت التبعيات الأساسية
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. تثبيت حزم البايثون الأساسية أولاً
RUN pip install --no-cache-dir sentence-transformers

# 3. تنزيل وتثبيت Ollama
RUN wget https://ollama.com/download/ollama-linux-amd64 -O /usr/bin/ollama \
    && chmod +x /usr/bin/ollama

# 4. تهيئة مجلد العمل ونسخ الملفات
WORKDIR /app
COPY . .

# 5. تثبيت متطلبات المشروع
RUN pip install --no-cache-dir -r requirements.txt

# 6. تعيين متغيرات البيئة
ENV OLLAMA_HOST=0.0.0.0:11434
ENV PYTHONUNBUFFERED=1

# 7. إنشاء مجلد للسجلات
RUN mkdir -p /var/log

# 8. نسخ ملف التشغيل بدلاً من إنشائه (أفضل ممارسة)
COPY setup.sh /app/setup.sh
RUN chmod +x /app/setup.sh

# 9. فحص الصحة (اختياري)
HEALTHCHECK --interval=30s --timeout=30s \
  CMD curl -f http://localhost:11434 || exit 1

# 10. تشغيل التطبيق
CMD ["/app/setup.sh"]