#!/bin/bash

# بدء خدمة Ollama في الخلفية
ollama serve &

# انتظر حتى تصبح الخدمة جاهزة
sleep 3

# تنزيل النموذج المطلوب
ollama pull deepseek-r1:1.5b

# بدء تطبيق Streamlit
streamlit run app.py --server.port=8501 --server.address=0.0.0.0