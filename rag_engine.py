import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import json
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Retrieval-Augmented Generation движок
    Хранит документы в векторной базе и ищет релевантные
    """

    def __init__(self, embedding_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Инициализация с моделью эмбеддингов
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.index = None
        self.documents = []
        self.metadata = []
        self.index_path = "vector_store/faiss.index"
        self.doc_path = "vector_store/documents.pkl"

        # Создаем папку для векторного хранилища
        os.makedirs("vector_store", exist_ok=True)

        # Загружаем существующий индекс если есть
        self.load_index()

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """
        Добавляет документы в векторную базу
        """
        try:
            # Создаем эмбеддинги для документов
            embeddings = self.embedding_model.encode(documents, show_progress_bar=True)

            # Если индекса нет, создаем новый
            if self.index is None:
                dimension = embeddings.shape[1]
                self.index = faiss.IndexFlatL2(dimension)

            # Добавляем эмбеддинги в индекс
            self.index.add(embeddings.astype('float32'))

            # Сохраняем документы и метаданные
            self.documents.extend(documents)
            if metadata:
                self.metadata.extend(metadata)
            else:
                self.metadata.extend([{} for _ in documents])

            # Сохраняем индекс
            self.save_index()

            logger.info(f"✅ Добавлено {len(documents)} документов в RAG базу")

        except Exception as e:
            logger.error(f"Ошибка добавления документов: {e}")

    def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """
        Ищет релевантные документы по запросу
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        try:
            # Создаем эмбеддинг для запроса
            query_embedding = self.embedding_model.encode([query])

            # Ищем ближайшие векторы
            distances, indices = self.index.search(query_embedding.astype('float32'), k)

            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.documents):
                    results.append({
                        'document': self.documents[idx],
                        'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                        'score': float(distances[0][i])
                    })

            return results

        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
            return []

    def add_faqs_from_json(self, json_path: str):
        """
        Загружает FAQ из JSON и добавляет в RAG базу
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            documents = []
            metadata = []

            for intent_name, intent_data in data.items():
                for pattern in intent_data.get('patterns', []):
                    # Добавляем вопрос
                    documents.append(pattern)
                    metadata.append({
                        'type': 'question',
                        'intent': intent_name
                    })

                for response in intent_data.get('responses', []):
                    # Добавляем ответ
                    documents.append(response)
                    metadata.append({
                        'type': 'answer',
                        'intent': intent_name
                    })

            self.add_documents(documents, metadata)
            logger.info(f"✅ Добавлено {len(documents)} записей из FAQ в RAG")

        except Exception as e:
            logger.error(f"Ошибка загрузки FAQ в RAG: {e}")

    def add_text_file(self, file_path: str, file_type: str = 'txt'):
        """
        Добавляет текстовый файл в базу знаний
        """
        try:
            if file_type == 'txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Разбиваем на чанки (по предложениям)
                chunks = self._split_into_chunks(text)

                documents = chunks
                metadata = [{'source': file_path, 'chunk': i} for i in range(len(chunks))]

                self.add_documents(documents, metadata)

            elif file_type == 'json':
                self.add_faqs_from_json(file_path)

            else:
                logger.warning(f"Неподдерживаемый тип файла: {file_type}")

        except Exception as e:
            logger.error(f"Ошибка добавления файла {file_path}: {e}")

    def _split_into_chunks(self, text: str, chunk_size: int = 200) -> List[str]:
        """
        Разбивает текст на чанки
        """
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def save_index(self):
        """
        Сохраняет индекс и документы
        """
        try:
            if self.index is not None:
                faiss.write_index(self.index, self.index_path)

            with open(self.doc_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)

            logger.info("✅ RAG индекс сохранен")

        except Exception as e:
            logger.error(f"Ошибка сохранения индекса: {e}")

    def load_index(self):
        """
        Загружает индекс и документы
        """
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)

            if os.path.exists(self.doc_path):
                with open(self.doc_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.metadata = data['metadata']

                logger.info(f"✅ RAG индекс загружен: {len(self.documents)} документов")

        except Exception as e:
            logger.error(f"Ошибка загрузки индекса: {e}")

    def get_context_for_query(self, query: str, max_chunks: int = 3) -> str:
        """
        Возвращает контекст для запроса (для передачи в LLM)
        """
        results = self.search(query, k=max_chunks)

        if not results:
            return ""

        context = "Вот информация из базы знаний, которая может помочь ответить на вопрос:\n\n"

        for i, result in enumerate(results, 1):
            context += f"[{i}] {result['document']}\n"
            if result['metadata']:
                context += f"   (источник: {result['metadata'].get('source', 'база знаний')})\n"
            context += "\n"

        context += "На основе этой информации дай ответ пользователю."
        return context