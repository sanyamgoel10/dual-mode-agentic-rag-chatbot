import pytest
import os

os.environ.setdefault("LLM_PROVIDER", "openai")
os.environ.setdefault("LLM_MODEL", "gpt-4o")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_API_KEY", "test-key")
os.environ.setdefault("PINECONE_INDEX", "test-index")
os.environ.setdefault("CURRENT_DATE", "2026-06-15")
os.environ.setdefault("DATABASE_PATH", "/tmp/test_orders.db")
os.environ.setdefault("ORDERS_CSV_PATH", "tests/fixtures/orders_sample.csv")
