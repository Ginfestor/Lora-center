import gradio as gr
import threading
import os
import shutil
import requests
from app.trainer import train_lora, log_path
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data/uploads"
ADAPTER_DIR = "adapters"
MODELS_DIR = "models"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Список моделей для загрузки
DOWNLOADABLE_MODELS = {
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Mistral": "mistralai/Mistral-7B-v0.1",
    "Falcon": "tiiuae/falcon-7b"
}

def check_model_availability(model_name):
    """Проверяет доступность модели в локальной директории"""
    model_path = os.path.join(MODELS_DIR, model_name)
    return os.path.exists(model_path)

def get_available_models():
    """Получает список доступных моделей"""
    available_models = {}
    for name in DOWNLOADABLE_MODELS.keys():
        if check_model_availability(name):
            available_models[name] = name
    return available_models

def download_model(model_name, model_id):
    """Загружает модель"""
    try:
        logger.info(f"Начало загрузки модели {model_name}")
        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(os.path.join(MODELS_DIR, model_name))
        
        # Загрузка модели
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        model.save_pretrained(os.path.join(MODELS_DIR, model_name))
        
        logger.info(f"Модель {model_name} успешно загружена")
        return f"✅ Модель {model_name} успешно загружена"
    except Exception as e:
        error_msg = f"❌ Ошибка при загрузке модели {model_name}: {str(e)}"
        logger.error(error_msg)
        return error_msg

def save_file(file):
    filepath = os.path.join(UPLOAD_DIR, file.name)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file, f)
    return f"Загружен файл: {file.name}"

def start_training(model_name, r, alpha, dropout):
    train_lora(model_name=model_name, r=int(r), alpha=int(alpha), dropout=float(dropout))
    return "✅ Обучение запущено"

def get_logs():
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return f.read()
    return "Логи пока пусты"

def chatbot(msg):
    return f"(бот отвечает): {msg}"

def create_ui():
    with gr.Blocks(title="LoRA Center") as demo:
        gr.Markdown("# Центр обучения LoRA-ботов")

        with gr.Tabs():
            with gr.Tab("📥 Загрузка моделей"):
                gr.Markdown("### Доступные модели для загрузки")
                model_download = gr.Dropdown(
                    label="Выберите модель для загрузки",
                    choices=list(DOWNLOADABLE_MODELS.keys()),
                    interactive=True
                )
                download_button = gr.Button("Загрузить модель")
                download_status = gr.Textbox(label="Статус загрузки")
                
                def handle_download(model_name):
                    if model_name:
                        model_id = DOWNLOADABLE_MODELS[model_name]
                        return download_model(model_name, model_id)
                    return "Пожалуйста, выберите модель"
                
                download_button.click(
                    handle_download,
                    inputs=model_download,
                    outputs=download_status
                )

            with gr.Tab("📤 Загрузка файлов"):
                upload = gr.File(file_types=[".txt", ".docx", ".pdf"], file_count="multiple")
                upload_button = gr.Button("Загрузить")
                output = gr.Textbox()
                upload_button.click(fn=lambda files: "\n".join([save_file(f) for f in files]), inputs=upload, outputs=output)

            with gr.Tab("🧠 Обучение LoRA"):
                with gr.Row():
                    with gr.Column():
                        available_models = get_available_models()
                        model_selector = gr.Dropdown(
                            label="Выберите модель",
                            choices=list(available_models.keys()),
                            value=None,
                            interactive=True
                        )
                        model_name = gr.Textbox(
                            label="Или введите HuggingFace ID",
                            value="",
                            interactive=True
                        )
                        def sync_model(choice):
                            return available_models.get(choice, "")
                        model_selector.change(sync_model, inputs=model_selector, outputs=model_name)
                    
                    with gr.Column():
                        r = gr.Number(label="r", value=8)
                        alpha = gr.Number(label="alpha", value=32)
                        dropout = gr.Number(label="dropout", value=0.05)
                
                train_button = gr.Button("Запустить обучение")
                train_output = gr.Textbox()
                train_button.click(start_training, inputs=[model_name, r, alpha, dropout], outputs=train_output)

            with gr.Tab("📊 Логи обучения"):
                log_output = gr.Textbox(label="Логи", lines=20)
                refresh_button = gr.Button("Обновить логи")
                refresh_button.click(get_logs, outputs=log_output)
                # Автоматическое обновление логов каждые 5 секунд
                demo.load(get_logs, outputs=log_output, every=5)

            with gr.Tab("🤖 Чат с ботом"):
                chat = gr.ChatInterface(fn=chatbot)

    return demo

def launch_ui():
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)