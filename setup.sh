#!/bin/bash

# بدء Ollama مع إعادة تحميل النموذج إذا لزم الأمر
start_ollama() {
    echo "Starting Ollama server..."
    ollama serve > /var/log/ollama.log 2>&1 &
    
    local max_retries=10
    local retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -s http://localhost:11434 >/dev/null; then
            echo "Ollama server started successfully"
            
            # التحقق من وجود النموذج
            if ! ollama list | grep -q "deepseek-r1:1.5b"; then
                echo "Pulling deepseek-r1:1.5b model..."
                ollama pull deepseek-r1:1.5b
            fi
            
            return 0
        fi
        
        echo "Waiting for Ollama to start... (Attempt $((retry_count+1))/$max_retries)"
        sleep 5
        ((retry_count++))
    done
    
    echo "Failed to start Ollama after $max_retries attempts"
    return 1
}

# بدء Streamlit
start_streamlit() {
    echo "Starting Streamlit application..."
    streamlit run app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --logger.level=debug
}

# التنظيف عند الإغلاق
cleanup() {
    echo "Cleaning up..."
    pkill -f "ollama serve"
}

# تنفيذ الرئيسي
main() {
    trap cleanup EXIT
    
    if start_ollama; then
        start_streamlit
    else
        echo "Failed to start the application due to Ollama service issues"
        exit 1
    fi
}

main