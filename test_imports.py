try:
    from telegram import Update
    print("✅ telegram-bot установлен")
except ImportError as e:
    print(f"❌ telegram-bot: {e}")

try:
    import aiohttp
    print("✅ aiohttp установлен")
except ImportError as e:
    print(f"❌ aiohttp: {e}")

try:
    from sentence_transformers import SentenceTransformer
    print("✅ sentence-transformers установлен")
except ImportError as e:
    print(f"❌ sentence-transformers: {e}")

try:
    import faiss
    print("✅ faiss установлен")
except ImportError as e:
    print(f"❌ faiss: {e}")

try:
    import sklearn
    print("✅ scikit-learn установлен")
except ImportError as e:
    print(f"❌ scikit-learn: {e}")

try:
    import numpy
    print("✅ numpy установлен")
except ImportError as e:
    print(f"❌ numpy: {e}")