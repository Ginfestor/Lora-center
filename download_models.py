from transformers import AutoModelForCausalLM, AutoTokenizer
import os

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Список моделей для загрузки
MODELS = {
    "TinyLlama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "Mistral": "mistralai/Mistral-7B-v0.1",
    "Falcon": "tiiuae/falcon-7b"
}

def download_model(model_name, model_id):
    print(f"Загрузка модели {model_name}...")
    try:
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
        
        print(f"✅ Модель {model_name} успешно загружена")
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели {model_name}: {str(e)}")

if __name__ == "__main__":
    print("Начинаем загрузку моделей...")
    for model_name, model_id in MODELS.items():
        download_model(model_name, model_id)
    print("Загрузка моделей завершена") 