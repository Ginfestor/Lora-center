import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import threading

# Настройка логирования
log_path = "training.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

MODELS_DIR = "models"

def train_lora(model_name, r=8, alpha=32, dropout=0.05):
    try:
        logging.info(f"Начало обучения LoRA для модели {model_name}")
        
        # Загрузка модели и токенизатора из локальной директории
        model_path = os.path.join(MODELS_DIR, model_name)
        if not os.path.exists(model_path):
            raise ValueError(f"Модель {model_name} не найдена в директории {MODELS_DIR}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Настройка LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout
        )
        
        model = get_peft_model(model, peft_config)
        
        # Загрузка датасета
        dataset = load_dataset("text", data_files={"train": "data/uploads/*.txt"})
        
        # Настройка аргументов обучения
        training_args = TrainingArguments(
            output_dir="adapters",
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            save_strategy="epoch"
        )
        
        # Создание тренера
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            data_collator=lambda data: {'input_ids': torch.stack([f['input_ids'] for f in data]),
                                      'attention_mask': torch.stack([f['attention_mask'] for f in data])}
        )
        
        # Запуск обучения в отдельном потоке
        def train_thread():
            trainer.train()
            trainer.save_model()
            logging.info("Обучение завершено")
        
        thread = threading.Thread(target=train_thread)
        thread.start()
        
        return "Обучение запущено в фоновом режиме"
        
    except Exception as e:
        logging.error(f"Ошибка при обучении: {str(e)}")
        raise