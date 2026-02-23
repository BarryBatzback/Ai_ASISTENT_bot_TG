import json
import numpy as np
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleNeuralBot:
    """
    Простая нейросеть для классификации интентов и обучения на диалогах
    """

    def __init__(self, model_path="models/simple_nn.pkl"):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            analyzer='char_wb'  # Учитывает русские слова лучше
        )
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.model_path = model_path
        self.intents = {}
        self.responses = {}

    def load_intents(self, json_path):
        """
        Загружает интенты из JSON файла
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            patterns = []
            intent_labels = []
            self.intents = {}
            self.responses = {}

            for intent_name, intent_data in data.items():
                # Сохраняем интенты
                intent_id = len(self.intents)
                self.intents[intent_id] = intent_name
                self.responses[intent_name] = intent_data.get('responses', [])

                # Добавляем паттерны
                for pattern in intent_data.get('patterns', []):
                    patterns.append(pattern.lower())
                    intent_labels.append(intent_name)

            logger.info(f"Загружено {len(patterns)} паттернов для {len(self.intents)} интентов")
            return patterns, intent_labels

        except Exception as e:
            logger.error(f"Ошибка загрузки интентов: {e}")
            return [], []

    def train(self, json_path):
        """
        Обучает нейросеть на JSON файле
        """
        try:
            # Загружаем данные
            patterns, intent_labels = self.load_intents(json_path)

            if not patterns:
                logger.error("Нет данных для обучения")
                return False

            # Преобразуем тексты в векторы
            X = self.vectorizer.fit_transform(patterns).toarray()

            # Кодируем метки
            y = self.label_encoder.fit_transform(intent_labels)

            # Обучаем классификатор
            self.classifier.fit(X, y)
            self.is_trained = True

            # Сохраняем модель
            self.save_model()

            logger.info(f"✅ Модель обучена на {len(patterns)} примерах")
            return True

        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            return False

    def predict(self, text):
        """
        Предсказывает интент для текста
        """
        if not self.is_trained:
            return None, 0.0

        try:
            # Векторизуем текст
            X = self.vectorizer.transform([text.lower()]).toarray()

            # Получаем вероятности для всех классов
            proba = self.classifier.predict_proba(X)[0]

            # Находим класс с максимальной вероятностью
            max_idx = np.argmax(proba)
            confidence = proba[max_idx]

            # Если уверенность太低, возвращаем None
            if confidence < 0.3:
                return None, confidence

            # Декодируем интент
            intent = self.label_encoder.inverse_transform([max_idx])[0]
            return intent, confidence

        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return None, 0.0

    def get_response(self, intent):
        """
        Возвращает случайный ответ для интента
        """
        if intent in self.responses and self.responses[intent]:
            return np.random.choice(self.responses[intent])
        return None

    def learn_from_dialog(self, user_message, bot_response, intent=None):
        """
        Обучается на новом диалоге
        """
        try:
            # Сохраняем диалог для дообучения
            dialog_file = "training_data/new_examples.json"

            # Загружаем существующие данные
            if os.path.exists(dialog_file):
                with open(dialog_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = []

            # Добавляем новый пример
            data.append({
                "pattern": user_message,
                "response": bot_response,
                "intent": intent
            })

            # Сохраняем
            with open(dialog_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            # Если накопилось много примеров, дообучаем модель
            if len(data) >= 100:
                self.retrain_on_new_data()

        except Exception as e:
            logger.error(f"Ошибка обучения на диалоге: {e}")

    def retrain_on_new_data(self):
        """
        Дообучает модель на новых данных
        """
        try:
            # Загружаем новые примеры
            with open("training_data/new_examples.json", 'r', encoding='utf-8') as f:
                new_data = json.load(f)

            if len(new_data) < 10:
                return

            # Загружаем исходные интенты
            patterns, intent_labels = self.load_intents("knowledge_base/faqs.json")

            # Добавляем новые примеры
            for item in new_data:
                if item.get('intent'):
                    patterns.append(item['pattern'])
                    intent_labels.append(item['intent'])

            # Дообучаем модель
            X = self.vectorizer.fit_transform(patterns).toarray()
            y = self.label_encoder.fit_transform(intent_labels)

            # Частичное обучение (если поддерживается)
            if hasattr(self.classifier, 'partial_fit'):
                self.classifier.partial_fit(X, y, classes=np.unique(y))
            else:
                self.classifier.fit(X, y)

            logger.info(f"✅ Модель дообучена на {len(new_data)} новых примерах")

            # Очищаем файл новых примеров
            with open("training_data/new_examples.json", 'w', encoding='utf-8') as f:
                json.dump([], f)

        except Exception as e:
            logger.error(f"Ошибка дообучения: {e}")

    def save_model(self):
        """
        Сохраняет модель в файл
        """
        try:
            os.makedirs("models", exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'classifier': self.classifier,
                    'label_encoder': self.label_encoder,
                    'intents': self.intents,
                    'responses': self.responses
                }, f)
            logger.info(f"✅ Модель сохранена в {self.model_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения модели: {e}")

    def load_model(self):
        """
        Загружает модель из файла
        """
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    data = pickle.load(f)

                self.vectorizer = data['vectorizer']
                self.classifier = data['classifier']
                self.label_encoder = data['label_encoder']
                self.intents = data['intents']
                self.responses = data['responses']
                self.is_trained = True

                logger.info(f"✅ Модель загружена из {self.model_path}")
                return True
            else:
                logger.warning("Модель не найдена")
                return False
        except Exception as e:
            logger.error(f"Ошибка загрузки модели: {e}")
            return False