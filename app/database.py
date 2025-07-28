import pymysql
import pandas as pd
from typing import Dict, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from app.config import settings

# SQLAlchemy 설정
SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{settings.DB_USERNAME}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

    # 데이터 추출 (SQLAlchemy 방식)
    # try:
    #     with SessionLocal() as db:
    #         select_query = text('''SELECT 
    #                             c.id,
    #                             c.overview,
    #                             c.title,
    #                             c.type,
    #                             c.release_date,
    #                             c.poster_path,
    #                             GROUP_CONCAT(g.name ORDER BY g.name SEPARATOR ', ') AS `genres`
    #                         FROM `contents` c
    #                         JOIN `content_genres` cg ON c.id = cg.content_id
    #                             AND c.type = cg.content_type
    #                         JOIN `genres` g ON cg.genre_id = g.id
    #                         GROUP BY 
    #                             c.id,
    #                             c.overview,
    #                             c.title,
    #                             c.release_date,
    #                             c.type,
    #                             c.poster_path''')
            
    #         result = db.execute(select_query).fetchall()
            
    #         # DataFrame 변환
    #         df = pd.DataFrame(result, columns=['id', 'overview', 'title', 'type', 'release_date', 'poster_path', 'genres'])
    #         print(df[:10])
    #         df.to_csv('data_processing/content_data.csv', index=False, encoding='utf-8-sig')

    # except Exception as e:
    #     print(f"Error: {e}")