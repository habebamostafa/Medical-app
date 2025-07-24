FROM python:3.9-slim

# 1. تثبيت جميع التبعيات النظامية دفعة واحدة
RUN apt-get update && apt-get install -y \
    wget \
    cmake \
    build-essential \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# 2. تثبيت SentencePiece من المصدر
RUN git clone https://github.com/google/sentencepiece.git \
    && cd sentencepiece \
    && mkdir build \
    && cd build \
    && cmake .. \
    && make -j $(nproc) \
    && make install \
    && ldconfig

# 3. تثبيت Ollama
RUN wget https://ollama.com/download/ollama-linux-amd64 -O /usr/bin/ollama \
    && chmod +x /usr/bin/ollama

# 4. إعداد بيئة العمل
WORKDIR /app
COPY . .

# 5. تثبيت متطلبات Python مع تحديث pip أولاً
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir sentencepiece protobuf

# 6. تعيين متغيرات البيئة
ENV OLLAMA_HOST=0.0.0.0:11434
ENV PYTHONUNBUFFERED=1

# 7. جعل ملف الإعداد قابلاً للتنفيذ
RUN chmod +x /app/setup.sh

# 8. اختبار تثبيت SentencePiece
RUN python -c "import sentencepiece; print('SentencePiece installed successfully')"

CMD ["/app/setup.sh"]