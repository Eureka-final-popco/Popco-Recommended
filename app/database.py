import mysql.connector
from mysql.connector import Error
from typing import List, Dict, Optional
from datetime import date
from sqlalchemy.exc import OperationalError
from config import settings
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

# SQLAlchemy 설정
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def check_db_connection():
    """
    SQLAlchemy engine을 사용하여 데이터베이스 연결을 확인합니다.
    """
    try:
        # engine.connect()는 실제로 DB에 연결을 시도합니다.
        connection = engine.connect()
        print("✅ 데이터베이스 연결에 성공했습니다.")
        print("연결된 DB URL : " + settings.DB_HOST)
        # 확인 후에는 반드시 연결을 닫아줍니다.
        connection.close()
        return True
    except OperationalError as e:
        # 연결 실패 시 (예: 잘못된 비밀번호, DB 서버 다운 등) OperationalError가 발생합니다.
        print(f"❌ 데이터베이스 연결에 실패했습니다: {e}")
        return False
    except Exception as e:
        # 그 외 예상치 못한 다른 오류 처리
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")
        return False

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
