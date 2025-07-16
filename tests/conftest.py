import os
import pytest

@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """테스트 환경변수 설정"""
    os.environ.setdefault("DB_HOST", "localhost")
    os.environ.setdefault("DB_PORT", "5432")
    os.environ.setdefault("DB_USERNAME", "test")
    os.environ.setdefault("DB_PASSWORD", "test")
    os.environ.setdefault("DB_NAME", "test_db")