import pymysql
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from config import settings

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
try:
    with SessionLocal() as db:
        select_query = text('''SELECT 
                            c.id,
                            c.overview,
                            c.title,
                            c.type,
                            GROUP_CONCAT(g.name ORDER BY g.name SEPARATOR ', ') AS `genres`
                        FROM `content` c
                        JOIN `content_genre_ids` cg ON c.id = cg.content_id
                        JOIN `genre` g ON cg.genre_id = g.id
                        GROUP BY 
                            c.id,
                            c.overview,
                            c.title,
                            c.release_date,
                            c.type''')
        
        result = db.execute(select_query).fetchall()
        
        # DataFrame 변환
        df = pd.DataFrame(result, columns=['id', 'overview', 'title', 'release_date', 'type', 'genres'])
        print(df[:10])
        df.to_csv('data_processing/content_data.csv', index=False, encoding='utf-8-sig')

except Exception as e:
    print(f"Error: {e}")