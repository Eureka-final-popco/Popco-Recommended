# main.py
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from config import settings
from database import get_db
import os

app = FastAPI(title="Simple FastAPI", version="1.0.0")

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
                "genres": result.genres  # 이제 genres 필드가 있음
            }
            return {"status": "Database connection successful", "data": result_dict}
    except Exception as e:
        return {"status": "Database connection failed", "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)