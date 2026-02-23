import os
import logging
import json
import sqlite3
import aiohttp
import asyncio
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, MessageHandler,
    filters, ContextTypes, CallbackQueryHandler
)

# Импортируем наши модули
from rag_engine import RAGEngine
from simple_nn import SimpleNeuralBot

# Загружаем переменные окружения
load_dotenv()

# ============================================
# НАСТРОЙКИ
# ============================================

BOT_TOKEN = os.getenv("8687116910:AAEBckqEQHOjRJ4B1hptLqw353tTwjgEAlM")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ============================================
# ИНИЦИАЛИЗАЦИЯ КОМПОНЕНТОВ
# ============================================

# RAG движок
rag_engine = RAGEngine()

# Простая нейросеть
simple_nn = SimpleNeuralBot()


# База данных для хранения диалогов
class DialogDatabase:
    def __init__(self, db_path="conversations.db"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.create_tables()

    def create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                user_name TEXT,
                user_message TEXT,
                bot_response TEXT,
                intent TEXT,
                timestamp DATETIME
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS user_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                rating INTEGER,
                feedback TEXT,
                timestamp DATETIME
            )
        """)
        self.conn.commit()

    def save_conversation(self, user_id, user_name, user_message, bot_response, intent=None):
        cursor = self.conn.execute(
            "INSERT INTO conversations (user_id, user_name, user_message, bot_response, intent, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, user_name, user_message, bot_response, intent, datetime.now())
        )
        self.conn.commit()
        return cursor.lastrowid

    def save_feedback(self, conversation_id, rating, feedback=""):
        self.conn.execute(
            "INSERT INTO user_feedback (conversation_id, rating, feedback, timestamp) VALUES (?, ?, ?, ?)",
            (conversation_id, rating, feedback, datetime.now())
        )
        self.conn.commit()

    def get_user_stats(self, user_id):
        cursor = self.conn.execute(
            "SELECT COUNT(*) FROM conversations WHERE user_id = ?",
            (user_id,)
        )
        return cursor.fetchone()[0]


# Создаем базу данных
db = DialogDatabase()

# Состояния игр для пользователей
user_games = {}


# Класс для игры "Угадай число"
class GuessNumberGame:
    def __init__(self):
        self.secret_number = random.randint(1, 100)
        self.attempts = 0
        self.max_attempts = 10
        self.is_active = True

    def guess(self, number):
        self.attempts += 1
        if number == self.secret_number:
            self.is_active = False
            return f"🎉 Поздравляю! Ты угадал число {self.secret_number} за {self.attempts} попыток!"
        elif number < self.secret_number:
            return f"📈 Загаданное число БОЛЬШЕ {number}. Осталось попыток: {self.max_attempts - self.attempts}"
        else:
            return f"📉 Загаданное число МЕНЬШЕ {number}. Осталось попыток: {self.max_attempts - self.attempts}"


# Класс для игры "Камень-ножницы-бумага"
class RPSGame:
    def __init__(self):
        self.choices = ["камень", "ножницы", "бумага"]
        self.user_score = 0
        self.bot_score = 0

    def play(self, user_choice):
        bot_choice = random.choice(self.choices)

        if user_choice == bot_choice:
            result = "🤝 Ничья!"
        elif (user_choice == "камень" and bot_choice == "ножницы") or \
                (user_choice == "ножницы" and bot_choice == "бумага") or \
                (user_choice == "бумага" and bot_choice == "камень"):
            result = "✅ Ты выиграл!"
            self.user_score += 1
        else:
            result = "❌ Я выиграл!"
            self.bot_score += 1

        return {
            "user": user_choice,
            "bot": bot_choice,
            "result": result,
            "scores": f"Счет: Ты {self.user_score} : {self.bot_score} Я"
        }


# ============================================
# ФУНКЦИИ ДЛЯ РАБОТЫ С OLLAMA
# ============================================

async def query_ollama(prompt: str, context: str = "", history: List[Dict] = None) -> str:
    """
    Отправляет запрос к Ollama
    """
    try:
        # Формируем системный промпт
        system_prompt = """Ты дружелюбный помощник по имени МегаБот. Твои особенности:
- Отвечаешь кратко и по делу (максимум 3-4 предложения)
- Используешь эмодзи для эмоций
- Ты вежливый и позитивный
- Если есть информация из базы знаний - используй её
- Отвечаешь на русском языке"""

        # Добавляем контекст из RAG если есть
        if context:
            system_prompt += f"\n\n{context}"

        # Формируем сообщения
        messages = [{"role": "system", "content": system_prompt}]

        # Добавляем историю если есть
        if history:
            messages.extend(history[-5:])  # Последние 5 сообщений

        # Добавляем текущий запрос
        messages.append({"role": "user", "content": prompt})

        # Отправляем запрос
        async with aiohttp.ClientSession() as session:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 500
                }
            }

            async with session.post(f"{OLLAMA_HOST}/api/chat", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("message", {}).get("content", "Извини, я не смог сгенерировать ответ.")
                else:
                    error_text = await response.text()
                    logger.error(f"Ошибка Ollama: {response.status} - {error_text}")
                    return "🚫 Ошибка связи с ИИ. Проверь, запущен ли Ollama."

    except Exception as e:
        logger.error(f"Исключение при запросе к Ollama: {e}")
        return f"😕 Произошла ошибка: {str(e)}"


# ============================================
# ОБРАБОТЧИКИ КОМАНД
# ============================================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Главное меню"""
    user = update.effective_user

    # Проверяем статистику
    msg_count = db.get_user_stats(user.id)

    keyboard = [
        [InlineKeyboardButton("🤖 Обычный чат", callback_data='chat')],
        [InlineKeyboardButton("📚 Спросить с RAG", callback_data='rag_chat')],
        [
            InlineKeyboardButton("🌤️ Погода", callback_data='weather'),
            InlineKeyboardButton("💵 Курс валют", callback_data='currency'),
        ],
        [
            InlineKeyboardButton("🔤 Переводчик", callback_data='translate'),
            InlineKeyboardButton("🎮 Игры", callback_data='games'),
        ],
        [
            InlineKeyboardButton("📊 Статистика", callback_data='stats'),
            InlineKeyboardButton("🧹 Очистить историю", callback_data='clear'),
        ],
        [InlineKeyboardButton("❓ Помощь", callback_data='help')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"🌟 Привет, {user.first_name}! 🌟\n\n"
        f"Я супер-бот с ИИ, RAG и обучением!\n"
        f"Отправлено сообщений: {msg_count}\n\n"
        f"Выбери режим работы:",
        reply_markup=reply_markup
    )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик нажатий на кнопки"""
    query = update.callback_query
    user_id = update.effective_user.id
    await query.answer()

    if query.data == 'chat':
        context.user_data['mode'] = 'chat'
        await query.edit_message_text(
            "🤖 **Режим обычного чата**\n\n"
            "Просто пиши мне сообщения, и я буду отвечать как ИИ!\n"
            "Я запоминаю историю нашего разговора.\n\n"
            "Для возврата в меню нажми /start",
            parse_mode='Markdown'
        )

    elif query.data == 'rag_chat':
        context.user_data['mode'] = 'rag'
        await query.edit_message_text(
            "📚 **Режим с RAG (поиск по документам)**\n\n"
            "Я буду искать ответы в своей базе знаний и дополнять их ИИ!\n"
            "Задавай любые вопросы.\n\n"
            "Для возврата в меню нажми /start",
            parse_mode='Markdown'
        )

    elif query.data == 'weather':
        await query.edit_message_text(
            "🌤️ **Узнать погоду**\n\n"
            "Напиши название города, например:\n"
            "`погода Москва`\n"
            "`погода Лондон`\n"
            "`погода Нью-Йорк`",
            parse_mode='Markdown'
        )

    elif query.data == 'currency':
        # Простые курсы валют
        rates = {
            'USD': 91.5,
            'EUR': 99.2,
            'GBP': 116.8,
            'JPY': 0.62,
            'CNY': 12.7
        }

        text = "💵 **Курсы валют к рублю**\n\n"
        for currency, rate in rates.items():
            text += f"• {currency}: {rate} ₽\n"
        text += "\n*Данные примерные. Для реальных курсов нужен API*"

        await query.edit_message_text(text, parse_mode='Markdown')

    elif query.data == 'translate':
        await query.edit_message_text(
            "🔤 **Переводчик**\n\n"
            "Я перевожу текст с русского на английский и обратно!\n\n"
            "Напиши:\n"
            "`переведи привет` - перевод на английский\n"
            "`translate hello` - перевод на русский",
            parse_mode='Markdown'
        )

    elif query.data == 'games':
        keyboard = [
            [
                InlineKeyboardButton("🎯 Угадай число", callback_data='game_guess'),
                InlineKeyboardButton("✂️ Камень-ножницы", callback_data='game_rps'),
            ],
            [InlineKeyboardButton("🔙 Назад", callback_data='back_to_menu')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🎮 **Выбери игру**\n\n"
            "🎯 Угадай число - я загадаю число от 1 до 100\n"
            "✂️ Камень-ножницы-бумага - классическая игра",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif query.data == 'game_guess':
        user_games[user_id] = {'game': 'guess', 'instance': GuessNumberGame()}
        await query.edit_message_text(
            "🎯 **Игра 'Угадай число'**\n\n"
            "Я загадал число от 1 до 100.\n"
            f"У тебя 10 попыток.\n\n"
            f"Напиши число:"
        )

    elif query.data == 'game_rps':
        user_games[user_id] = {'game': 'rps', 'instance': RPSGame()}
        keyboard = [
            [
                InlineKeyboardButton("🪨 Камень", callback_data='rps_rock'),
                InlineKeyboardButton("✂️ Ножницы", callback_data='rps_scissors'),
                InlineKeyboardButton("📄 Бумага", callback_data='rps_paper'),
            ],
            [InlineKeyboardButton("🚪 Выйти из игры", callback_data='games')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "✂️ **Камень-ножницы-бумага**\n\n"
            "Выбери свой ход:",
            reply_markup=reply_markup
        )

    elif query.data.startswith('rps_'):
        if user_id not in user_games or user_games[user_id]['game'] != 'rps':
            await query.edit_message_text("Игра не найдена. Начни новую игру.")
            return

        game = user_games[user_id]['instance']
        choice_map = {
            'rps_rock': 'камень',
            'rps_scissors': 'ножницы',
            'rps_paper': 'бумага'
        }

        user_choice = choice_map[query.data]
        result = game.play(user_choice)

        await query.edit_message_text(
            f"🤖 **Результат:**\n\n"
            f"Ты: {result['user']}\n"
            f"Я: {result['bot']}\n"
            f"{result['result']}\n\n"
            f"{result['scores']}\n\n"
            f"Хочешь сыграть еще?",
            reply_markup=InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("🪨 Камень", callback_data='rps_rock'),
                    InlineKeyboardButton("✂️ Ножницы", callback_data='rps_scissors'),
                    InlineKeyboardButton("📄 Бумага", callback_data='rps_paper'),
                ],
                [InlineKeyboardButton("🚪 Выйти", callback_data='games')],
            ])
        )

    elif query.data == 'stats':
        msg_count = db.get_user_stats(user_id)
        await query.edit_message_text(
            f"📊 **Твоя статистика**\n\n"
            f"Всего сообщений: {msg_count}\n"
            f"Режим: {context.user_data.get('mode', 'не выбран')}\n\n"
            f"Используй /start для возврата в меню",
            parse_mode='Markdown'
        )

    elif query.data == 'clear':
        if 'history' in context.user_data:
            context.user_data['history'] = []
        await query.edit_message_text(
            "🧹 **История диалога очищена!**\n\n"
            "Начинаем с чистого листа.",
            parse_mode='Markdown'
        )

    elif query.data == 'help':
        keyboard = [[InlineKeyboardButton("🔙 В меню", callback_data='back_to_menu')]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "❓ **Помощь**\n\n"
            "📌 **Режимы работы:**\n"
            "• Обычный чат - просто общение с ИИ\n"
            "• RAG чат - поиск по документам + ИИ\n\n"
            "📌 **Команды:**\n"
            "/start - главное меню\n"
            "/help - эта справка\n"
            "/train - обучение на диалогах\n"
            "/feedback - оставить отзыв\n\n"
            "📌 **Игры:**\n"
            "В меню 'Игры' доступны:\n"
            "- Угадай число\n"
            "- Камень-ножницы-бумага",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    elif query.data == 'back_to_menu':
        keyboard = [
            [InlineKeyboardButton("🤖 Обычный чат", callback_data='chat')],
            [InlineKeyboardButton("📚 Спросить с RAG", callback_data='rag_chat')],
            [
                InlineKeyboardButton("🌤️ Погода", callback_data='weather'),
                InlineKeyboardButton("💵 Курс валют", callback_data='currency'),
            ],
            [
                InlineKeyboardButton("🔤 Переводчик", callback_data='translate'),
                InlineKeyboardButton("🎮 Игры", callback_data='games'),
            ],
            [
                InlineKeyboardButton("📊 Статистика", callback_data='stats'),
                InlineKeyboardButton("🧹 Очистить историю", callback_data='clear'),
            ],
            [InlineKeyboardButton("❓ Помощь", callback_data='help')],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await query.edit_message_text(
            "🌟 **Главное меню** 🌟\n\n"
            "Выбери режим работы:",
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )


async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для обучения бота"""
    user_id = update.effective_user.id

    # Проверяем, есть ли данные для обучения
    cursor = db.conn.execute(
        "SELECT user_message, bot_response, intent FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT 50",
        (user_id,)
    )
    conversations = cursor.fetchall()

    if len(conversations) < 5:
        await update.message.reply_text(
            "📚 Для обучения нужно больше диалогов. Напиши со мной хотя бы 5-10 сообщений."
        )
        return

    await update.message.reply_text(
        "🧠 **Начинаю обучение на твоих диалогах...**\n"
        "Это может занять несколько секунд.",
        parse_mode='Markdown'
    )

    # Обучаем нейросеть на диалогах пользователя
    for conv in conversations:
        simple_nn.learn_from_dialog(conv[0], conv[1], conv[2])

    await update.message.reply_text(
        "✅ **Обучение завершено!**\n"
        "Теперь я буду лучше понимать тебя.",
        parse_mode='Markdown'
    )


async def feedback_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда для обратной связи"""
    await update.message.reply_text(
        "📝 **Оставь отзыв**\n\n"
        "Напиши свой отзыв о моей работе. Например:\n"
        "`отзыв 5 Бот супер!`\n"
        "где 5 - оценка от 1 до 5.\n\n"
        "Или просто напиши 'отзыв' и я покажу последние отзывы."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик всех сообщений"""
    user_text = update.message.text
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name

    # Проверяем команды обучения и отзывов
    if user_text.startswith('отзыв'):
        parts = user_text.split()
        if len(parts) >= 2:
            try:
                rating = int(parts[1])
                feedback = ' '.join(parts[2:]) if len(parts) > 2 else ""

                # Сохраняем отзыв
                last_conv = db.conn.execute(
                    "SELECT id FROM conversations WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1",
                    (user_id,)
                ).fetchone()

                if last_conv:
                    db.save_feedback(last_conv[0], rating, feedback)
                    await update.message.reply_text(
                        f"✅ Спасибо за отзыв! Оценка: {rating}/5"
                    )
                else:
                    await update.message.reply_text("Сначала напиши что-нибудь, чтобы я мог оценить ответ.")
            except ValueError:
                await update.message.reply_text("Используй формат: отзыв 5 Твой комментарий")
        else:
            # Показываем последние отзывы
            cursor = db.conn.execute(
                "SELECT rating, feedback, timestamp FROM user_feedback ORDER BY timestamp DESC LIMIT 5"
            )
            feedbacks = cursor.fetchall()

            if feedbacks:
                text = "📊 **Последние отзывы:**\n\n"
                for fb in feedbacks:
                    text += f"• Оценка: {fb[0]}/5\n"
                    if fb[1]:
                        text += f"  Комментарий: {fb[1]}\n"
                    text += f"  {fb[2][:16]}\n\n"
                await update.message.reply_text(text, parse_mode='Markdown')
            else:
                await update.message.reply_text("Пока нет отзывов. Будь первым!")
        return

    # Проверяем запрос погоды
    if user_text.lower().startswith('погода'):
        city = user_text[6:].strip()
        if city:
            weather_data = get_weather(city)
            await update.message.reply_text(weather_data)
            db.save_conversation(user_id, user_name, user_text, weather_data, 'weather')
        else:
            await update.message.reply_text("Напиши название города, например: погода Москва")
        return

    # Проверяем перевод
    if user_text.lower().startswith('переведи'):
        text = user_text[7:].strip()
        if text:
            translation = translate_text(text)
            await update.message.reply_text(translation)
            db.save_conversation(user_id, user_name, user_text, translation, 'translate')
        else:
            await update.message.reply_text("Напиши что перевести, например: переведи привет")
        return

    if user_text.lower().startswith('translate'):
        text = user_text[9:].strip()
        if text:
            translation = translate_to_russian(text)
            await update.message.reply_text(translation)
            db.save_conversation(user_id, user_name, user_text, translation, 'translate')
        else:
            await update.message.reply_text("Write what to translate, for example: translate hello")
        return

    # Проверяем игру
    if user_id in user_games:
        game_data = user_games[user_id]

        if game_data['game'] == 'guess':
            try:
                number = int(user_text)
                game = game_data['instance']
                result = game.guess(number)

                if not game.is_active:
                    del user_games[user_id]
                    await update.message.reply_text(result)
                    db.save_conversation(user_id, user_name, user_text, result, 'game')
                else:
                    await update.message.reply_text(result)
                    db.save_conversation(user_id, user_name, user_text, result, 'game')
            except ValueError:
                await update.message.reply_text("Пожалуйста, введи число от 1 до 100!")
            return

    # Если нет специального режима, используем ИИ
    mode = context.user_data.get('mode', 'chat')

    # Показываем, что бот думает
    await update.message.chat.send_action(action="typing")

    # Проверяем, может ли простая нейросеть ответить
    intent, confidence = simple_nn.predict(user_text)

    if intent and confidence > 0.7:
        # Если нейросеть уверена, используем её ответ
        response = simple_nn.get_response(intent)
        if response:
            await update.message.reply_text(response)
            db.save_conversation(user_id, user_name, user_text, response, intent)

            # Обучаем на этом диалоге
            simple_nn.learn_from_dialog(user_text, response, intent)
            return

    # Если нейросеть не уверена, используем Ollama + RAG
    if mode == 'rag':
        # Ищем в RAG базе
        rag_context = rag_engine.get_context_for_query(user_text)

        # Получаем историю из контекста пользователя
        history = context.user_data.get('history', [])

        # Отправляем запрос в Ollama с контекстом из RAG
        response = await query_ollama(user_text, rag_context, history)
    else:
        # Обычный чат без RAG
        history = context.user_data.get('history', [])
        response = await query_ollama(user_text, "", history)

    await update.message.reply_text(response)

    # Сохраняем в историю
    if 'history' not in context.user_data:
        context.user_data['history'] = []

    context.user_data['history'].append({"role": "user", "content": user_text})
    context.user_data['history'].append({"role": "assistant", "content": response})

    # Ограничиваем историю
    if len(context.user_data['history']) > 20:
        context.user_data['history'] = context.user_data['history'][-20:]

    # Сохраняем в базу данных
    db.save_conversation(user_id, user_name, user_text, response, intent if intent else 'ai')

    # Обучаем простую нейросеть на этом диалоге
    simple_nn.learn_from_dialog(user_text, response, intent)


# ============================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================

def get_weather(city):
    """Получение погоды (упрощенная версия)"""
    import random

    weather_types = ["☀️ ясно", "⛅ облачно", "☁️ пасмурно", "🌧️ дождь", "🌨️ снег", "🌩️ гроза"]
    temperatures = [-5, 0, 5, 10, 15, 20, 25]

    weather = random.choice(weather_types)
    temp = random.choice(temperatures)
    humidity = random.randint(40, 90)
    wind = random.randint(1, 10)

    return (
        f"🌍 **Погода в {city.title()}**\n\n"
        f"{weather}\n"
        f"🌡️ Температура: {temp}°C\n"
        f"💧 Влажность: {humidity}%\n"
        f"💨 Ветер: {wind} м/с\n\n"
        f"*Данные примерные для демонстрации*"
    )


def translate_text(text):
    """Простой перевод с русского на английский"""
    translations = {
        "привет": "hello",
        "как дела": "how are you",
        "пока": "goodbye",
        "спасибо": "thank you",
        "доброе утро": "good morning",
        "добрый вечер": "good evening",
        "да": "yes",
        "нет": "no",
        "я тебя люблю": "i love you",
        "как тебя зовут": "what is your name",
        "сколько времени": "what time is it",
        "где": "where",
        "почему": "why",
        "кто": "who",
        "что": "what"
    }

    text_lower = text.lower()
    for ru, en in translations.items():
        if ru in text_lower:
            return f"🔤 **Перевод:** '{text}'\n→ '{en}'"

    return f"🔤 **Примерный перевод:** '{text}' → *перевод в разработке*"


def translate_to_russian(text):
    """Простой перевод с английского на русский"""
    translations = {
        "hello": "привет",
        "how are you": "как дела",
        "goodbye": "пока",
        "thank you": "спасибо",
        "good morning": "доброе утро",
        "good evening": "добрый вечер",
        "yes": "да",
        "no": "нет",
        "i love you": "я тебя люблю",
        "what is your name": "как тебя зовут",
        "what time is it": "сколько времени",
        "where": "где",
        "why": "почему",
        "who": "кто",
        "what": "что"
    }

    text_lower = text.lower()
    for en, ru in translations.items():
        if en in text_lower:
            return f"🔤 **Translation:** '{text}'\n→ '{ru}'"

    return f"🔤 **Примерный перевод:** '{text}' → *перевод в разработке*"


# ============================================
# ЗАПУСК БОТА
# ============================================

async def post_init(application: Application):
    """Действия после инициализации бота"""
    # Загружаем базу знаний в RAG
    if os.path.exists("knowledge_base/faqs.json"):
        rag_engine.add_faqs_from_json("knowledge_base/faqs.json")

    # Обучаем простую нейросеть
    if simple_nn.load_model() is False:
        if os.path.exists("knowledge_base/faqs.json"):
            simple_nn.train("knowledge_base/faqs.json")

    logger.info("✅ Бот инициализирован и готов к работе!")


def main():
    """Запуск бота"""
    print("=" * 60)
    print("🚀 Запуск СУПЕР-БОТА (Ollama + RAG + Нейросеть)...")
    print("=" * 60)

    # Проверяем наличие токена
    if not BOT_TOKEN:
        print("❌ Ошибка: Не найден BOT_TOKEN в файле .env")
        return

    try:
        # Создаем приложение
        application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()

        # Добавляем обработчики команд
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", button_callback))
        application.add_handler(CommandHandler("train", train_command))
        application.add_handler(CommandHandler("feedback", feedback_command))

        # Добавляем обработчик кнопок
        application.add_handler(CallbackQueryHandler(button_callback))

        # Добавляем обработчик сообщений
        application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

        print("✅ Бот успешно запущен!")
        print("📱 Открой Telegram и начни общение")
        print("🛑 Для остановки нажми Ctrl+C")
        print("=" * 60)

        # Запускаем бота
        application.run_polling(allowed_updates=Update.ALL_TYPES)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        logger.error(f"Ошибка: {e}")


if __name__ == "__main__":
    import random

    main()