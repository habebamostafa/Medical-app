# FROM python:3.9-slim

# # تثبيت التبعيات الأساسية
# RUN apt-get update && apt-get install -y \
#     wget \
#     curl \
#     && rm -rf /var/lib/apt/lists/*

# # إنشاء مجلد للسجلات
# RUN mkdir -p /var/log

# # نسخ الملفات المطلوبة أولاً (لتحسين caching)
# WORKDIR /app
# COPY requirements.txt .
# COPY app.py .
# COPY setup.sh .

# # تثبيت متطلبات Python
# RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir sentence-transformers

# # تثبيت Ollama
# RUN wget https://ollama.com/download/ollama-linux-amd64 -O /usr/bin/ollama \
#     && chmod +x /usr/bin/ollama

# # تعيين متغيرات البيئة
# ENV OLLAMA_HOST=127.0.0.1:11434
# ENV PYTHONUNBUFFERED=1

# # جعل ملف التشغيل قابل للتنفيذ
# RUN chmod +x /app/setup.sh

# # فحص الصحة
# HEALTHCHECK --interval=30s --timeout=30s \
#   CMD curl -f http://127.0.0.1:11434 || exit 1

# # تشغيل التطبيق
# CMD ["/app/setup.sh"]