import os
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import load_dataset
import threading
import time

# Настройка логирования
log_path = "training.log"
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Абсолютный путь к директории моделей в корне проекта
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

def train_lora(model_name, r=8, alpha=32, dropout=0.05):
    try:
        logging.info(f"Начало обучения LoRA для модели {model_name}")
        logging.info(f"Используется GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"Доступная память GPU: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Проверка наличия файлов для обучения
        upload_dir = "data/uploads"
        txt_files = [f for f in os.listdir(upload_dir) if f.endswith('.txt') and os.path.isfile(os.path.join(upload_dir, f))]
        if not txt_files:
            logging.error("Нет файлов для обучения в директории data/uploads")
            return "Ошибка: нет файлов для обучения. Загрузите текстовые файлы (.txt) во вкладке 'Обучение LoRA'"
        
        logging.info(f"Найдено файлов для обучения: {len(txt_files)}")
        
        # Загрузка модели и токенизатора из локальной директории
        model_path = os.path.join(MODELS_DIR, model_name.replace('/', '_'))
        if not os.path.exists(model_path):
            raise ValueError(f"Модель {model_name} не найдена в директории {MODELS_DIR}")
        
        # Проверяем, содержит ли путь к модели подпапки с моделью
        model_config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(model_config_path):
            # Ищем подкаталоги с файлом config.json
            subdirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(model_path, subdir)
                if os.path.exists(os.path.join(subdir_path, "config.json")):
                    logging.info(f"Найдена модель в подкаталоге: {subdir}")
                    model_path = subdir_path
                    break
            else:
                raise ValueError(f"Не найден файл config.json для модели {model_name}")
        
        logging.info(f"Загрузка модели из {model_path}")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            max_memory={0: "4GB"},  # Уменьшаем до 4GB для GPU с 7.66GB памяти
            low_cpu_mem_usage=True,  # Оптимизация использования CPU памяти
            offload_folder="offload"  # Папка для выгрузки на CPU
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
        dataset = load_dataset("text", data_files={"train": os.path.join(upload_dir, "*.txt")})
        
        # Проверка размера датасета
        if len(dataset["train"]) == 0:
            logging.error("Датасет пуст после загрузки")
            return "Ошибка: датасет пуст. Проверьте, что файлы содержат текст"
        
        logging.info(f"Размер датасета: {len(dataset['train'])} примеров")
        
        # Токенизация данных
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")
        
        # Токенизируем датасет
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
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
            train_dataset=tokenized_dataset["train"],
            data_collator=lambda data: {
                'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in data]),
                'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in data]),
                'labels': torch.stack([torch.tensor(f['input_ids']) for f in data])
            }
        )
        
        # Запуск обучения в отдельном потоке
        def train_thread():
            try:
                trainer.train()
                # Сохранение адаптера с именем модели и датой
                save_path = os.path.join("adapters", f"{model_name.replace('/', '_')}-{int(time.time())}")
                trainer.save_model(save_path)
                logging.info(f"Обучение завершено, модель сохранена в {save_path}")
            except Exception as e:
                logging.error(f"Ошибка в процессе обучения: {str(e)}")
        
        thread = threading.Thread(target=train_thread)
        thread.start()
        
        return "Обучение запущено в фоновом режиме. Проверьте вкладку 'Логи обучения' для отслеживания прогресса."
        
    except Exception as e:
        error_msg = f"Ошибка при обучении: {str(e)}"
        logging.error(error_msg)
        return error_msg