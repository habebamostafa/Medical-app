# 🩺 AI-Powered Medical Assistant Bot

A powerful multilingual chatbot for medical question answering, OCR extraction, and drug review analysis — built using `LangChain`, `Ollama`, `PaddleOCR`, `Telegram Bot API`, and more.

---

##  Features

- **Conversational AI**: Uses LLMs (e.g., DeepSeek, Gemma) via Ollama for accurate, context-based medical answers.
- **Drug Review Dataset**: Analyzes real user reviews from the `"lewtun/drug-reviews"` dataset.
- **Context Memory**: Remembers previous interactions and follows up on patient queries.
- **Arabic & English Support**: Translates between Arabic and English for seamless multilingual communication.
- **PDF/Photo OCR**: Extracts text from prescriptions, labels, and documents using `PaddleOCR` and `PyMuPDF`.
- **Telegram Bot Integration**: Query using text, images, or PDF documents.
- **Safety Focused**: Includes warnings and safe practice guidelines.

---

## Requirements

Install dependencies with:

```bash
pip install \
  faiss-cpu python-dotenv telebot nltk paddleocr paddlepaddle \
  langchain langchain_community datasets fsspec \
  langdetect sentence-transformers pytesseract \
  python-telegram-bot nest_asyncio PyMuPDF \
  gradio transformers
