import os
import sys

from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from contextlib import asynccontextmanager
import pandas as pd
import logging

from app.config import settings
from app.database import get_db

from app.persona_based_recommender.state import pbr_app_state
from app.persona_based_recommender.persona_router import persona_recommender_router
from app.persona_based_recommender.data_loader import load_all_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("메인 FastAPI 애플 시작: 모든 추천 시스템 데이터 초기화 시작.")
    
    try:
        # data_loader의 load_all_data 함수를 직접 호출
        # 이제 db_url 인자를 전달할 필요가 없습니다.
        load_all_data() 
        logger.info("추천 시스템 데이터 초기화 완료.")
        yield
    except Exception as e:
        logger.error(f"추천 시스템 초기화 오류 발생: {e}", exc_info=True)
        raise RuntimeError("추천 시스템 초기화 실패") from e


app = FastAPI(title="POPCO Recommendation API", version="1.0.0", lifespan=lifespan)

@app.get("/")
async def root():
    return {
        "message": "Hello World from FastAPI!",
        "status": "running",
        "environment": "docker" if os.getenv("DOCKER_ENV") else "local"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/test/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id, "message": f"This is item {item_id}"}

@app.get("/db-test")
def test_db(db: Session = Depends(get_db)):
    try:
        query = text("""
            SELECT
                c.id, c.title, c.overview, c.type,
                GROUP_CONCAT(g.name ORDER BY g.name SEPARATOR ', ') AS genres
            FROM content c
            LEFT JOIN content_genre_ids cg ON c.id = cg.content_id
            LEFT JOIN genre g ON cg.genre_id = g.id
            WHERE c.id = 11
            GROUP BY c.id, c.title, c.overview, c.type
        """)

        result = db.execute(query).first()
        if result:
            result_dict = {
                "id": result.id,
                "title": result.title,
                "overview": result.overview,
                "type": result.type,
                "genres": result.genres
            }
            return {"status": "Database connection successful", "data": result_dict}
        else:
            return {"status": "Database connection successful", "data": "No content found for ID 11."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

app.include_router(persona_recommender_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)