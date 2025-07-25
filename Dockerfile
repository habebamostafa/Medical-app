FROM python:3.9-slim

# تثبيت التبعيات الأساسية
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# تثبيت sentence-transformers أولاً (يتطلب بناءً بعض الحزم)
RUN pip install --no-cache-dir sentence-transformers

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
ENV PYTHONUNBUFFERED=1

# إنشاء ملف setup.sh قابل للتنفيذ
RUN echo '#!/bin/bash\n\
ollama serve &\n\
sleep 5\n\
ollama pull deepseek-r1:1.5b\n\
streamlit run app.py --server.port=8501 --server.address=0.0.0.0\n\
' > /app/setup.sh && \
    chmod +x /app/setup.sh
RUN ollama serve & \
    sleep 10 && \
    ollama list && \
    pkill ollama
# تشغيل البرنامج النصي للإعداد
CMD ["/app/setup.sh"]