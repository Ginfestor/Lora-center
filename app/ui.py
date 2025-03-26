import os
import logging
import gradio as gr
import threading
import shutil
import requests
from app.trainer import train_lora, log_path
import time
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi, login, HfFolder
from dotenv import load_dotenv
from peft import PeftModel

# Загружаем переменные окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = "data/uploads"
ADAPTER_DIR = "adapters"
# Абсолютный путь к директории моделей в корне проекта
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(ADAPTER_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Получаем токен из переменных окружения
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    logger.warning("HF_TOKEN не найден в переменных окружения")

# Инициализируем пустой словарь моделей
DOWNLOADABLE_MODELS = {}

# Кэш для загруженных моделей и токенизаторов
MODEL_CACHE = {}
TOKENIZER_CACHE = {}

def get_popular_models(hf_token=None):
    """Получает список популярных моделей с HuggingFace"""
    try:
        logger.info("Начало получения списка моделей")
        api = HfApi()
        
        # Используем токен из переменных окружения, если не передан
        token = hf_token or HF_TOKEN
        if token:
            logger.info("Сохраняем токен HuggingFace")
            HfFolder.save_token(token)
            logger.info("Токен успешно сохранен")
        
        logger.info("Запрашиваем модели с HuggingFace")
        # Получаем популярные модели для текстовой генерации
        try:
            models = api.list_models(
                filter="text-generation",
                sort="downloads",
                direction=-1,
                limit=50,  # Увеличиваем лимит
                full=True,  # Получаем полную информацию о моделях
                cardData=True  # Получаем метаданные из карточек моделей
            )
            logger.info("Запрос к API успешно выполнен")
        except Exception as api_error:
            logger.error(f"Ошибка при запросе к API: {str(api_error)}")
            logger.exception("Полный стек ошибки API:")
            return {}
        
        models_list = list(models)
        logger.info(f"Получено {len(models_list)} моделей")
        
        # Отладка: печатаем атрибуты первой модели
        if models_list and len(models_list) > 0:
            first_model = models_list[0]
            logger.info(f"Пример модели (первая в списке): {first_model.id}")
            
            # Выводим все доступные атрибуты
            logger.info("Доступные атрибуты модели:")
            for attr_name in dir(first_model):
                if not attr_name.startswith("_"):  # Пропускаем приватные атрибуты
                    try:
                        attr_value = getattr(first_model, attr_name)
                        # Если это не метод, выводим значение
                        if not callable(attr_value):
                            if attr_name == "cardData" and attr_value:
                                logger.info(f"cardData структура:")
                                for key, value in attr_value.items():
                                    logger.info(f"  - {key}: {type(value)}")
                                    # Для model-index выводим подробнее
                                    if key == "model-index" and isinstance(value, list) and value:
                                        logger.info(f"    model-index[0] keys: {value[0].keys() if isinstance(value[0], dict) else 'не словарь'}")
                            else:
                                logger.info(f"  - {attr_name}: {attr_value}")
                    except Exception as e:
                        logger.info(f"  - {attr_name}: <ошибка доступа: {str(e)}>")
        
        # Формируем словарь моделей
        model_dict = {}
        for model in models_list:
            try:
                # Проверяем, что модель поддерживает текстовую генерацию
                if hasattr(model, 'tags') and "text-generation" in model.tags:
                    # Собираем информацию о модели
                    model_info = {
                        'id': model.id,
                        'downloads': getattr(model, 'downloads', 0),
                        'likes': getattr(model, 'likes', 0),
                        'tags': getattr(model, 'tags', []),
                        'pipeline_tag': getattr(model, 'pipeline_tag', ''),
                        'description': getattr(model, 'description', 'Описание отсутствует'),
                        'cardData': getattr(model, 'cardData', {})
                    }
                    # Используем id модели как ключ
                    model_dict[model.id] = model_info
                    logger.info(f"Добавлена модель: {model.id} (загрузок: {model_info['downloads']})")
            except Exception as model_error:
                logger.error(f"Ошибка при обработке модели {model.id}: {str(model_error)}")
                continue
        
        logger.info(f"Итого добавлено {len(model_dict)} моделей")
        logger.info(f"Список моделей: {list(model_dict.keys())}")
        return model_dict
    except Exception as e:
        logger.error(f"Ошибка при получении списка моделей: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return {}

def check_model_availability(model_name):
    """Проверяет доступность модели в локальной директории"""
    model_path = os.path.join(MODELS_DIR, model_name)
    return os.path.exists(model_path)

def get_available_models():
    """Получает список доступных моделей"""
    available_models = {}
    
    # Проверяем фактическое наличие моделей в директории
    logger.info(f"Поиск моделей в директории: {MODELS_DIR}")
    if os.path.exists(MODELS_DIR):
        logger.info(f"Содержимое директории моделей: {os.listdir(MODELS_DIR)}")
        for model_dir in os.listdir(MODELS_DIR):
            model_path = os.path.join(MODELS_DIR, model_dir)
            if os.path.isdir(model_path):
                logger.info(f"Проверка директории модели: {model_dir}")
                # Рекурсивно ищем файлы модели
                has_model_files = False
                model_files = []
                for root, _, files in os.walk(model_path):
                    model_files.extend([f for f in files if f.endswith('.bin') or f.endswith('.safetensors')])
                    if any(file.endswith('.bin') or file.endswith('.safetensors') for file in files):
                        has_model_files = True
                        break
                
                logger.info(f"Найдены файлы модели: {model_files}")
                
                if has_model_files:
                    # Пытаемся прочитать информацию о модели из model_info.json
                    info_path = os.path.join(model_path, "model_info.json")
                    if os.path.exists(info_path):
                        try:
                            with open(info_path, "r", encoding="utf-8") as f:
                                import json
                                model_info = json.load(f)
                                model_name = model_info.get("model_name", model_dir)
                                model_id = model_info.get("model_id", model_dir)
                                available_models[model_name] = model_id
                                logger.info(f"Добавлена модель из model_info.json: {model_name} -> {model_id}")
                        except Exception as e:
                            logger.error(f"Ошибка при чтении model_info.json для {model_dir}: {str(e)}")
                            # Если не удалось прочитать информацию, используем имя директории
                            available_models[model_dir] = model_dir.replace('_', '/')
                            logger.info(f"Добавлена модель из имени директории: {model_dir} -> {model_dir.replace('_', '/')}")
                    else:
                        # Если нет файла с информацией, используем имя директории
                        available_models[model_dir] = model_dir.replace('_', '/')
                        logger.info(f"Добавлена модель из имени директории: {model_dir} -> {model_dir.replace('_', '/')}")
    
    logger.info(f"Итого найдено моделей: {len(available_models)}")
    logger.info(f"Доступные модели: {available_models}")
    return available_models

def update_transformers():
    """Обновляет библиотеку transformers до последней версии"""
    try:
        logger.info("Начало обновления библиотеки transformers")
        import subprocess
        
        # Сначала пробуем обновить через pip
        result = subprocess.run(
            ["pip", "install", "--upgrade", "transformers"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("Библиотека transformers успешно обновлена")
            return "✅ Библиотека transformers успешно обновлена. Попробуйте загрузить модель снова."
        else:
            logger.error(f"Ошибка при обновлении через pip: {result.stderr}")
            
            # Если обновление через pip не сработало, пробуем установить из репозитория
            logger.info("Попытка установки transformers из репозитория GitHub")
            result = subprocess.run(
                ["pip", "install", "git+https://github.com/huggingface/transformers.git"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logger.info("Библиотека transformers успешно установлена из GitHub")
                return "✅ Библиотека transformers успешно установлена из GitHub. Попробуйте загрузить модель снова."
            else:
                logger.error(f"Ошибка при установке из GitHub: {result.stderr}")
                return f"❌ Не удалось обновить библиотеку transformers. Попробуйте вручную выполнить: \n```\npip install --upgrade transformers\n```\nили\n```\npip install git+https://github.com/huggingface/transformers.git\n```"
    except Exception as e:
        logger.error(f"Ошибка при обновлении transformers: {str(e)}")
        return f"❌ Ошибка при обновлении: {str(e)}"

def download_model(model_name, model_id, hf_token):
    """Загружает модель"""
    try:
        logger.info(f"Начало загрузки модели {model_name}")
        
        # Создаем правильный путь для сохранения модели
        # Используем ID модели для создания структуры директорий
        model_path = os.path.join(MODELS_DIR, model_id.replace('/', '_'))
        os.makedirs(model_path, exist_ok=True)
        
        # Авторизация на HuggingFace
        if hf_token:
            login(token=hf_token)
            logger.info("Успешная авторизация на HuggingFace")
        
        # Сначала проверяем информацию о модели
        try:
            api = HfApi()
            model_info = api.model_info(
                repo_id=model_id,
                token=hf_token
            )
            
            # Проверяем, что модель поддерживает генерацию текста
            allowed_types = ['text-generation', 'text2text-generation', 'any-to-any', 'multi_modality']
            
            if hasattr(model_info, 'pipeline_tag') and model_info.pipeline_tag not in allowed_types:
                error_msg = f"❌ Модель {model_name} имеет тип {model_info.pipeline_tag}, но требуется модель для генерации текста"
                logger.error(error_msg)
                return error_msg
        
        except Exception as info_error:
            logger.warning(f"Не удалось проверить информацию о модели {model_name}: {str(info_error)}")
            logger.info("Продолжаем загрузку модели без проверки типа")
        
        # Загрузка токенизатора
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
            tokenizer.save_pretrained(model_path)
            logger.info(f"Токенизатор для модели {model_name} успешно загружен")
        except Exception as tokenizer_error:
            error_msg = f"❌ Ошибка при загрузке токенизатора: {str(tokenizer_error)}"
            logger.error(error_msg)
            return error_msg
        
        # Загрузка модели
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype="auto",
                device_map="auto",
                use_auth_token=hf_token
            )
            model.save_pretrained(model_path)
            logger.info(f"Модель {model_name} успешно загружена")
            
            # Создаем файл с метаданными модели
            with open(os.path.join(model_path, "model_info.json"), "w", encoding="utf-8") as f:
                import json
                json.dump({
                    "model_name": model_name,
                    "model_id": model_id,
                    "download_date": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, ensure_ascii=False, indent=2)
            
            return f"✅ Модель {model_name} успешно загружена"
        except Exception as model_error:
            error_msg = f"❌ Ошибка при загрузке модели {model_name}: {str(model_error)}"
            logger.error(error_msg)
            
            # Если загрузка модели не удалась, удаляем частично загруженные файлы
            try:
                shutil.rmtree(model_path)
                logger.info(f"Удалены частично загруженные файлы модели {model_name}")
            except Exception as cleanup_error:
                logger.error(f"Ошибка при удалении частично загруженных файлов: {str(cleanup_error)}")
            return error_msg
            
    except Exception as e:
        error_msg = f"❌ Ошибка при загрузке модели {model_name}: {str(e)}"
        logger.error(error_msg)
        return error_msg

def save_file(file):
    filepath = os.path.join(UPLOAD_DIR, file.name)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file, f)
    return f"Загружен файл: {file.name}"

def start_training(model, r, alpha, dropout):
    """Запускает процесс обучения LoRA"""
    # Проверяем, что модель указана
    if not model:
        return "Пожалуйста, выберите модель для обучения"
    
    # Если модель выбрана из выпадающего списка, используем её ID
    available_models = get_available_models()
    if model in available_models:
        model_id = available_models[model]
    else:
        # Иначе используем указанное значение напрямую
        model_id = model
    
    try:
        # Запускаем обучение в отдельном потоке
        train_thread = threading.Thread(
            target=train_lora,
            args=(model_id, r, alpha, dropout),
            daemon=True
        )
        train_thread.start()
        
        return f"Начато обучение модели {model_id}. Параметры: r={r}, alpha={alpha}, dropout={dropout}"
    except Exception as e:
        logger.error(f"Ошибка при запуске обучения: {str(e)}")
        return f"Ошибка при запуске обучения: {str(e)}"

def get_logs():
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return f.read()
    return "Логи пока пусты"

def chatbot(message, history, model_name, lora_selector="Нет"):
    """Взаимодействие с чат-ботом с использованием выбранной модели"""
    # Проверяем, что модель выбрана
    if not model_name:
        return "Пожалуйста, выберите модель в выпадающем списке"
    
    try:
        # Получаем список доступных моделей
        available_models = get_available_models()
        
        # Проверяем, что выбранная модель доступна
        if model_name not in available_models:
            return "Выбранная модель недоступна. Пожалуйста, выберите другую модель."
        
        # Получаем ID модели
        model_id = available_models[model_name]
        
        # Формируем путь к модели
        model_path = os.path.join(MODELS_DIR, model_id.replace("/", "_"))
        
        # Проверяем, что модель существует
        if not os.path.exists(model_path):
            return f"Модель {model_name} не найдена по пути {model_path}"
        
        # Проверяем, является ли модель директорией верхнего уровня или подкаталогом
        model_config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(model_config_path):
            # Ищем подкаталоги с файлом config.json
            subdirs = [d for d in os.listdir(model_path) if os.path.isdir(os.path.join(model_path, d))]
            for subdir in subdirs:
                subdir_path = os.path.join(model_path, subdir)
                if os.path.exists(os.path.join(subdir_path, "config.json")):
                    logger.info(f"Найдена модель в подкаталоге: {subdir}")
                    model_path = subdir_path
                    break
            else:
                return f"Не найден файл config.json для модели {model_name}"
        
        # Загружаем токенизатор (используем кэш)
        if model_name not in TOKENIZER_CACHE:
            logger.info(f"Загрузка токенизатора для модели {model_name} из {model_path}")
            TOKENIZER_CACHE[model_name] = AutoTokenizer.from_pretrained(model_path)
        tokenizer = TOKENIZER_CACHE[model_name]
        
        # Проверяем наличие LoRA адаптера
        use_lora = lora_selector != "Нет"
        lora_path = None
        
        if use_lora:
            # Получаем путь к LoRA адаптеру
            lora_path = os.path.join(ADAPTER_DIR, lora_selector)
            logger.info(f"Использование LoRA адаптера: {lora_selector}, путь: {lora_path}")
            
            # Проверяем существование директории адаптера
            if not os.path.exists(lora_path):
                return f"LoRA адаптер {lora_selector} не найден по пути {lora_path}"
            
            # Проверяем наличие необходимых файлов
            adapter_config = os.path.join(lora_path, "adapter_config.json")
            if not os.path.exists(adapter_config):
                return f"Не найден файл adapter_config.json для LoRA адаптера {lora_selector}"
        
        # Идентификатор для кэша модели (учитываем LoRA)
        cache_key = f"{model_name}_{lora_selector}" if use_lora else model_name
        
        # Загружаем модель (используем кэш)
        if cache_key not in MODEL_CACHE:
            logger.info(f"Загрузка модели {model_name} для чата из {model_path}")
            # Сначала загружаем базовую модель
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map="auto",
                local_files_only=True
            )
            
            # Если используется LoRA, загружаем адаптер
            if use_lora:
                try:
                    logger.info(f"Загрузка LoRA адаптера из {lora_path}")
                    # Загружаем LoRA адаптер
                    model = PeftModel.from_pretrained(base_model, lora_path)
                    logger.info(f"LoRA адаптер успешно загружен")
                except Exception as lora_error:
                    logger.error(f"Ошибка при загрузке LoRA адаптера: {str(lora_error)}")
                    return f"Ошибка при загрузке LoRA адаптера: {str(lora_error)}"
            else:
                model = base_model
            
            MODEL_CACHE[cache_key] = model
        else:
            logger.info(f"Использование кэшированной модели {cache_key}")
        
        model = MODEL_CACHE[cache_key]
        
        # Формируем промпт на основе истории чата
        prompt = ""
        for usr_msg, bot_msg in history:
            if usr_msg and bot_msg:
                prompt += f"Пользователь: {usr_msg}\nБот: {bot_msg}\n"
        
        # Добавляем текущее сообщение
        prompt += f"Пользователь: {message}\nБот:"
        
        # Генерируем ответ
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True
        )
        
        # Декодируем ответ
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Извлекаем только ответ бота
        try:
            # Пытаемся найти последний ответ бота
            bot_response = response.split("Бот:")[-1].strip()
        except:
            # Если не удалось разделить, используем весь ответ
            bot_response = response
        
        return bot_response
        
    except Exception as e:
        logger.error(f"Ошибка при генерации ответа: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return f"Произошла ошибка при генерации ответа: {str(e)}"

def handle_download(model_name, token):
    if model_name:
        # Используем полный ID модели из словаря
        model_info = DOWNLOADABLE_MODELS[model_name]
        model_id = model_info['id'] if isinstance(model_info, dict) else model_info
        return download_model(model_name, model_id, token)
    return "Пожалуйста, выберите модель"

def update_model_list(token):
    """Обновляет список моделей в интерфейсе"""
    global DOWNLOADABLE_MODELS
    try:
        logger.info("Начало обновления списка моделей")
        # Получаем модели с HuggingFace
        logger.info("Получаем популярные модели с HuggingFace")
        popular_models = get_popular_models(token)
        
        if popular_models:
            logger.info(f"Получено {len(popular_models)} моделей")
            # Обновляем глобальный словарь моделей
            DOWNLOADABLE_MODELS = popular_models
            choices = list(popular_models.keys())
            logger.info(f"Список моделей для выпадающего списка: {choices}")
            if not choices:
                logger.warning("Получен пустой список моделей")
            return choices
        else:
            logger.warning("Не удалось получить список моделей")
            DOWNLOADABLE_MODELS = {}  # Очищаем словарь моделей
            return []
            
    except Exception as e:
        logger.error(f"Ошибка при обновлении списка моделей: {str(e)}")
        logger.exception("Полный стек ошибки:")
        DOWNLOADABLE_MODELS = {}  # Очищаем словарь моделей
        return []

def get_model_info(model_id, hf_token=None):
    """Получает подробную информацию о конкретной модели с HuggingFace"""
    try:
        logger.info(f"Запрашиваем информацию о модели {model_id}")
        api = HfApi()
        
        # Используем токен из переменных окружения, если не передан
        token = hf_token or HF_TOKEN
        if token:
            HfFolder.save_token(token)
        
        # Получаем информацию о модели
        try:
            model_info = api.model_info(
                repo_id=model_id,
                token=token
            )
            logger.info(f"Успешно получена информация о модели {model_id}")
            
            # Преобразуем данные в словарь для удобства
            model_data = {
                'id': model_id,
                'downloads': getattr(model_info, 'downloads', 0),
                'likes': getattr(model_info, 'likes', 0),
                'tags': getattr(model_info, 'tags', []),
                'pipeline_tag': getattr(model_info, 'pipeline_tag', ''),
                'description': getattr(model_info, 'description', ''),
                'cardData': getattr(model_info, 'cardData', {})
            }
            
            # Отладка: выводим всю информацию о модели
            logger.info(f"Данные модели {model_id}:")
            for key, value in model_data.items():
                if key != 'cardData':  # Слишком много данных
                    logger.info(f"  - {key}: {value}")
            
            return model_data
            
        except Exception as api_error:
            logger.error(f"Ошибка при запросе информации о модели {model_id}: {str(api_error)}")
            logger.exception("Полный стек ошибки API:")
            return None
    except Exception as e:
        logger.error(f"Ошибка при получении информации о модели {model_id}: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return None

# Функция для получения краткого описания модели
def get_model_description(model_name):
    if not model_name or model_name not in DOWNLOADABLE_MODELS:
        return "Выберите модель для просмотра информации"
    
    # Получаем базовую информацию из кэша
    model_info = DOWNLOADABLE_MODELS[model_name]
    
    if isinstance(model_info, dict):
        # Пытаемся получить дополнительную информацию напрямую
        detailed_info = get_model_info(model_name)
        
        # Используем детальную информацию, если доступна
        if detailed_info:
            model_info = detailed_info
        
        # Получаем данные из модели
        downloads = model_info.get('downloads', 0)
        likes = model_info.get('likes', 0)
        tags = model_info.get('tags', [])
        card_data = model_info.get('cardData', {})
        
        # Пытаемся найти описание в разных местах
        description = ''
        
        # 1. Проверяем стандартное поле description
        if not description:
            description = model_info.get('description', '')
        
        # 2. Проверяем в cardData.model-index[0].description
        if not description and card_data:
            model_index = card_data.get('model-index', [])
            if model_index and isinstance(model_index, list) and len(model_index) > 0:
                first_entry = model_index[0]
                if isinstance(first_entry, dict) and 'description' in first_entry:
                    description = first_entry.get('description', '')
        
        # 3. Проверяем в cardData.metadata
        if not description and card_data:
            metadata = card_data.get('metadata', {})
            if isinstance(metadata, dict):
                description = metadata.get('description', '')
        
        # Если описание всё равно не найдено
        if not description:
            # Генерируем стандартное описание на основе имени и архитектуры
            description = f"Модель {model_name} для генерации текста."
            
            # Определяем архитектуру по имени
            architecture = None
            if "gpt" in model_name.lower():
                architecture = "GPT"
            elif "llama" in model_name.lower():
                architecture = "LLaMA"
            elif "mistral" in model_name.lower():
                architecture = "Mistral"
            elif "falcon" in model_name.lower():
                architecture = "Falcon"
            elif "phi" in model_name.lower():
                architecture = "Phi"
            
            if architecture:
                description += f" Основана на архитектуре {architecture}."
        
        # Извлекаем размер модели
        model_size = "Не указан"
        
        # 1. Сначала проверяем в метаданных карточки
        if card_data:
            # Проверяем поле model-index
            model_index = card_data.get('model-index', [])
            if model_index and isinstance(model_index, list) and len(model_index) > 0:
                # Извлекаем данные из первого элемента
                first_entry = model_index[0]
                if isinstance(first_entry, dict):
                    # Ищем параметры модели
                    params = first_entry.get('params', {})
                    if params:
                        param_count = params.get('n_params', None)
                        if param_count:
                            # Преобразуем в удобный формат (миллиарды или миллионы)
                            if param_count >= 1_000_000_000:
                                model_size = f"{param_count / 1_000_000_000:.1f}B параметров"
                            else:
                                model_size = f"{param_count / 1_000_000:.1f}M параметров"
        
        # 2. Если не нашли в метаданных, ищем в ID модели или тегах
        if model_size == "Не указан":
            # Ищем в ID модели
            size_patterns = [
                r'(\d+\.?\d*)b\b', # например, 7b, 7.5b
                r'(\d+\.?\d*)[_-]?b\b', # например, 7-b, 7_b
                r'(\d+\.?\d*)B\b', # например, 7B, 13B
                r'(\d+\.?\d*)[_-]?B\b', # например, 7-B, 7_B
                r'(\d+\.?\d*)[_-]?billion', # например, 7-billion
                r'(\d+\.?\d*)[_-]?bn\b', # например, 7bn
                r'-(\d+\.?\d*)b\b', # например, Llama-7b
                r'-(\d+\.?\d*)B\b' # например, Llama-7B
            ]
            
            # Ищем в ID модели
            for pattern in size_patterns:
                match = re.search(pattern, model_name, re.IGNORECASE)
                if match:
                    model_size = f"{match.group(1)}B параметров"
                    break
            
            # Если не нашли в ID, ищем в тегах
            if model_size == "Не указан" and tags:
                tags_str = " ".join(tags) if isinstance(tags, list) else str(tags)
                for pattern in size_patterns:
                    match = re.search(pattern, tags_str, re.IGNORECASE)
                    if match:
                        model_size = f"{match.group(1)}B параметров"
                        break
        
        # Обрезаем описание если оно слишком длинное
        short_desc = description[:300] + '...' if len(description) > 300 else description
        
        # Форматируем теги для отображения
        if isinstance(tags, list):
            tags_str = ', '.join(tags) if tags else "Нет тегов"
        else:
            tags_str = str(tags) if tags else "Нет тегов"
        
        # Формируем красивый информационный блок
        info_block = [
            f"**Модель:** {model_name}",
            f"**Размер модели:** {model_size}",
            f"**Описание:** {short_desc}",
            f"**Загрузки:** {downloads:,}",
            f"**Лайки:** {likes}"
        ]
        
        # Добавляем теги, только если их не слишком много
        if len(tags_str) < 500:
            info_block.append(f"**Теги:** {tags_str}")
        else:
            info_block.append(f"**Теги:** {len(tags) if isinstance(tags, list) else 'много'} тегов")
        
        return "\n\n".join(info_block)
    else:
        return f"**Модель:** {model_name}\n\nИнформация недоступна"

def create_ui():
    global DOWNLOADABLE_MODELS
    
    # Предварительно загружаем модели перед созданием интерфейса
    try:
        model_choices = update_model_list(HF_TOKEN)
        logger.info(f"Предварительно загружены модели: {model_choices}")
    except Exception as e:
        logger.error(f"Ошибка при предварительной загрузке моделей: {str(e)}")
        model_choices = []
    
    # Создаем интерфейс
    with gr.Blocks(title="LoRA Center") as demo:
        gr.Markdown("# Центр обучения LoRA-ботов")

        # Вкладка загрузки моделей
        with gr.Tab(label="📥 Загрузка моделей"):
            gr.Markdown("### Доступные модели для загрузки")
            hf_token = gr.Textbox(
                label="HuggingFace API токен",
                type="password",
                value=HF_TOKEN,
                placeholder="hf_..."
            )
            
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("#### Выбор из популярных моделей")
                    refresh_button = gr.Button("🔄 Обновить список моделей")
                    
                    # Важно: устанавливаем список выбора с предварительно загруженными моделями
                    model_download = gr.Dropdown(
                        label="Выберите модель для загрузки",
                        choices=model_choices,
                        interactive=True
                    )
                    
                with gr.Column(scale=2):
                    gr.Markdown("#### Прямой ввод ID модели")
                    custom_model_id = gr.Textbox(
                        label="Введите HuggingFace ID модели",
                        placeholder="например: mistralai/Mistral-7B-v0.1"
                    )
                    custom_download_button = gr.Button("Загрузить указанную модель")
            
            # Информация о модели
            model_info = gr.Markdown("Выберите модель для просмотра информации")
            
            with gr.Row():
                download_button = gr.Button("Загрузить выбранную модель")
                download_status = gr.Textbox(label="Статус загрузки")
            
            # Добавляем блок с обновлением библиотеки transformers
            with gr.Accordion("Дополнительные опции", open=False):
                gr.Markdown("#### Управление библиотеками")
                gr.Markdown("Если вы столкнулись с ошибкой загрузки модели из-за несовместимости архитектуры, попробуйте обновить библиотеку transformers:")
                update_transformers_button = gr.Button("🔄 Обновить transformers")
                transformers_status = gr.Textbox(label="Статус обновления")
                
                # Обработчик обновления transformers
                update_transformers_button.click(
                    fn=update_transformers,
                    inputs=[],
                    outputs=transformers_status
                )
            
            debug_output = gr.Textbox(label="Отладочная информация", visible=False)
            
            # Обработчик обновления списка моделей
            refresh_button.click(
                fn=update_model_list,
                inputs=hf_token,
                outputs=model_download
            )
            
            # Обработчик отображения информации о модели
            model_download.change(
                fn=get_model_description,
                inputs=model_download,
                outputs=model_info
            )
            
            # Обработчик загрузки выбранной модели
            download_button.click(
                fn=handle_download,
                inputs=[model_download, hf_token],
                outputs=download_status
            )
            
            # Обработчик загрузки модели по ID
            def download_custom_model(model_id, token):
                if not model_id or not model_id.strip():
                    return "Пожалуйста, введите ID модели"
                
                return download_model(model_id, model_id, token)
            
            # Отображение информации о введенной модели
            custom_model_id.change(
                fn=lambda model_id: get_model_description(model_id) if model_id and model_id.strip() else "Введите ID модели для просмотра информации",
                inputs=custom_model_id,
                outputs=model_info
            )
            
            # Запуск загрузки введенной модели
            custom_download_button.click(
                fn=download_custom_model,
                inputs=[custom_model_id, hf_token],
                outputs=download_status
            )

        # Остальные вкладки без изменений
        with gr.Tab(label="📤 Загрузка файлов"):
            upload = gr.File(file_types=[".txt", ".docx", ".pdf"], file_count="multiple")
            upload_button = gr.Button("Загрузить")
            output = gr.Textbox()
            upload_button.click(fn=lambda files: "\n".join([save_file(f) for f in files]), inputs=upload, outputs=output)

        with gr.Tab(label="🧠 Обучение LoRA"):
            gr.Markdown("### Загрузите текстовые данные для обучения")
            
            # Файловый загрузчик для данных обучения
            file_output = gr.File(
                file_count="multiple",
                label="Загрузите текстовые файлы для обучения (.txt)",
                file_types=[".txt"],
                type="filepath"
            )
            
            with gr.Row():
                upload_button = gr.Button("📤 Загрузить файлы")
                clear_files_button = gr.Button("🗑️ Очистить файлы")
            
            upload_info = gr.Markdown("")
            
            gr.Markdown("### Выберите модель и параметры обучения")
            
            # Компоненты для выбора модели и параметров
            with gr.Row():
                # Добавляем информационное сообщение для отображения статуса загрузки моделей
                model_status = gr.Markdown("Нажмите кнопку 'Обновить список моделей' для загрузки доступных моделей")
            
            with gr.Row():
                model_selector = gr.Dropdown(
                    label="Модель для обучения",
                    choices=[],  # Будет заполнено при обновлении
                    interactive=True
                )
                refresh_models_button = gr.Button("🔄 Обновить список моделей")
            
            # Параметры LoRA
            with gr.Row():
                lora_r = gr.Slider(label="Rank (r)", minimum=1, maximum=64, value=8, step=1)
                lora_alpha = gr.Slider(label="Alpha", minimum=1, maximum=64, value=32, step=1)
                lora_dropout = gr.Slider(label="Dropout", minimum=0.0, maximum=1.0, value=0.05, step=0.01)
            
            # Кнопка запуска обучения
            train_button = gr.Button("🚀 Запустить обучение")
            
            # Информация о процессе
            train_info = gr.Markdown("")
            
            # Загрузка файлов
            def upload_training_files(files):
                if not files:
                    return "Файлы не выбраны"
                
                uploaded_count = 0
                for file in files:
                    try:
                        # Получаем путь к файлу и проверяем его существование
                        file_path = file.name if hasattr(file, 'name') else file
                        if os.path.exists(file_path):
                            # Копируем файл в директорию для загрузок
                            destination = os.path.join(UPLOAD_DIR, os.path.basename(file_path))
                            shutil.copy2(file_path, destination)
                            logger.info(f"Файл {file_path} успешно скопирован в {destination}")
                            uploaded_count += 1
                        else:
                            logger.error(f"Файл {file_path} не найден")
                    except Exception as e:
                        logger.error(f"Ошибка при копировании файла: {str(e)}")
                
                # Дополнительная проверка загруженных файлов
                txt_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith('.txt') and os.path.isfile(os.path.join(UPLOAD_DIR, f))]
                logger.info(f"Всего файлов .txt в {UPLOAD_DIR}: {len(txt_files)}")
                
                if uploaded_count > 0:
                    return f"Загружено {uploaded_count} файлов в {UPLOAD_DIR}. Всего файлов в директории: {len(txt_files)}"
                else:
                    return "Ошибка: не удалось загрузить файлы. Проверьте лог."
            
            upload_button.click(
                fn=upload_training_files,
                inputs=file_output,
                outputs=upload_info
            )
            
            # Очистка файлов
            def clear_files():
                for file in os.listdir(UPLOAD_DIR):
                    file_path = os.path.join(UPLOAD_DIR, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                return "Все файлы удалены"
            
            clear_files_button.click(fn=clear_files, outputs=upload_info)
            
            # Обновление списка моделей
            def update_model_selector():
                models, status_text = refresh_train_models()
                # Используем gr.update() для обновления компонентов в Gradio 4.x
                return gr.update(choices=models, value=models[0] if models else None), status_text
                
            refresh_models_button.click(
                fn=update_model_selector,
                outputs=[model_selector, model_status]
            )
            
            # Запуск обучения
            train_button.click(
                fn=start_training,
                inputs=[model_selector, lora_r, lora_alpha, lora_dropout],
                outputs=train_info
            )

        with gr.Tab(label="📊 Логи обучения"):
            log_output = gr.Textbox(label="Логи", lines=20)
            refresh_button = gr.Button("Обновить логи")
            refresh_button.click(get_logs, outputs=log_output)
            # Автоматическое обновление логов каждые 5 секунд
            demo.load(get_logs, outputs=log_output, every=5)

        with gr.Tab(label="🤖 Чат с ботом"):
            with gr.Row():
                chat_model_selector = gr.Dropdown(
                    label="Выберите модель для чата",
                    choices=[],  # Будет заполнено при обновлении
                    interactive=True
                )
                refresh_chat_models_button = gr.Button("🔄 Обновить список моделей")
                clear_cache_button = gr.Button("🧹 Очистить кэш моделей")
            
            # Статус загрузки моделей для чата
            chat_model_status = gr.Markdown("Нажмите кнопку 'Обновить список моделей' для загрузки доступных моделей")
            
            # Добавляем выбор LoRA адаптера
            with gr.Row():
                lora_selector = gr.Dropdown(
                    label="LoRA адаптер (опционально)",
                    choices=["Нет"],  # Будет заполнено при обновлении
                    value="Нет",
                    interactive=True
                )
                refresh_loras_button = gr.Button("🔄 Обновить список LoRA адаптеров")
            
            # Статус загрузки LoRA адаптеров
            lora_status = gr.Markdown("Нажмите кнопку 'Обновить список LoRA адаптеров' для загрузки доступных адаптеров")
            
            # Обновляем список моделей для чата
            def update_chat_models():
                models, status_text = refresh_train_models()  # Используем ту же функцию, что и для обучения
                return gr.update(choices=models, value=models[0] if models else None), status_text
            
            refresh_chat_models_button.click(
                fn=update_chat_models,
                outputs=[chat_model_selector, chat_model_status]
            )
            
            # Обновление списка LoRA адаптеров
            def update_loras():
                loras = get_available_loras()
                lora_choices = ["Нет"] + [lora["name"] for lora in loras]
                status_text = f"Найдено {len(loras)} LoRA адаптеров"
                return gr.update(choices=lora_choices, value="Нет"), status_text
            
            refresh_loras_button.click(
                fn=update_loras,
                outputs=[lora_selector, lora_status]
            )
            
            # Очистка кэша моделей
            clear_cache_button.click(
                fn=lambda: clear_model_cache(),
                outputs=chat_model_status
            )
            
            # При изменении модели очищаем кэш, кроме выбранной модели
            chat_model_selector.change(
                fn=clear_model_cache,
                inputs=chat_model_selector,
                outputs=chat_model_status
            )
            
            # Интерфейс чата
            chat = gr.ChatInterface(
                fn=chatbot,
                additional_inputs=[chat_model_selector, lora_selector],
                title="Чат с ботом",
                description="Общайтесь с выбранной моделью. Не забудьте выбрать модель из списка выше!"
            )

    return demo

def launch_ui():
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

def refresh_train_models():
    """Обновляет список доступных моделей для обучения"""
    try:
        # Получаем список доступных моделей
        available_models = get_available_models()
        
        # Если список пуст, возвращаем сообщение об ошибке
        if not available_models:
            logger.warning("Не найдено моделей для обучения")
            return [], "Не найдено моделей для обучения. Сначала загрузите модель во вкладке 'Загрузка моделей'."
        
        # Возвращаем список моделей для выпадающего списка
        model_names = list(available_models.keys())
        logger.info(f"Обновление списка моделей для обучения: {model_names}")
        return model_names, f"Найдено {len(model_names)} моделей для обучения"
    except Exception as e:
        logger.error(f"Ошибка при обновлении списка моделей: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return [], f"Ошибка при обновлении списка моделей: {str(e)}"

def clear_model_cache(exclude_model=None):
    """Очищает кэш моделей, кроме исключенной модели"""
    global MODEL_CACHE, TOKENIZER_CACHE
    
    models_to_remove = [model for model in MODEL_CACHE.keys() if model != exclude_model]
    for model in models_to_remove:
        logger.info(f"Удаление модели {model} из кэша")
        del MODEL_CACHE[model]
    
    # Очищаем кэш токенизаторов, кроме исключенной модели
    tokenizers_to_remove = [t for t in TOKENIZER_CACHE.keys() if t != exclude_model]
    for t in tokenizers_to_remove:
        logger.info(f"Удаление токенизатора {t} из кэша")
        del TOKENIZER_CACHE[t]
    
    # Принудительно вызываем сборщик мусора
    import gc
    gc.collect()
    
    # Если есть доступ к CUDA, очищаем кэш
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA кэш очищен")
    except:
        pass
    
    return f"Кэш моделей очищен. Оставлена только модель {exclude_model if exclude_model else 'нет'}"

# Функция для получения доступных LoRA адаптеров
def get_available_loras():
    """Получает список доступных LoRA адаптеров"""
    available_loras = []
    lora_dir = ADAPTER_DIR
    
    logger.info(f"Поиск LoRA адаптеров в директории: {lora_dir}")
    if os.path.exists(lora_dir):
        # Получаем все директории в папке adapters, исключая checkpoint-*
        loras = [d for d in os.listdir(lora_dir) if os.path.isdir(os.path.join(lora_dir, d)) and not d.startswith("checkpoint-")]
        logger.info(f"Найдено LoRA адаптеров: {len(loras)}")
        logger.info(f"Список LoRA адаптеров: {loras}")
        
        for lora in loras:
            # Проверяем наличие необходимых файлов для LoRA адаптера
            lora_path = os.path.join(lora_dir, lora)
            adapter_config = os.path.join(lora_path, "adapter_config.json")
            adapter_model = os.path.join(lora_path, "adapter_model.bin")
            
            if os.path.exists(adapter_config) and (os.path.exists(adapter_model) or any(f.endswith(".bin") or f.endswith(".safetensors") for f in os.listdir(lora_path))):
                # Получаем имя базовой модели из имени адаптера
                base_model_name = lora.split("-")[0].replace("_", "/")
                available_loras.append({
                    "name": lora,
                    "path": lora_path,
                    "base_model": base_model_name
                })
    
    return available_loras

if __name__ == "__main__":
    launch_ui()