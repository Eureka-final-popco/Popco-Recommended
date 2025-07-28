from pydantic import BaseModel
from typing import Optional, List
from enum import Enum
from datetime import date
from fastapi import FastAPI

app = FastAPI()

class MovieRecommendation(BaseModel):
    content_id: int
    type: str
    title: str
    score: float
    poster_path: str
    genres: List[str]
    main_actors: List[str]
    directors: List[str]

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[MovieRecommendation]

#     # 추천 API 엔드포인트
# @app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
# async def get_user_recommendations(user_id: int, limit: int = 10):
#     try:
#         # 1. 사용자 선호도 계산
#         user_preferences = calculate_user_preferences(user_id)
        
#         # 2. 코사인 유사도 계산
#         raw_scores = calculate_cosine_similarities(user_preferences)
        
#         # 3. 상위 N개 추출
#         top_recommendations = raw_scores[:limit]
        
#         # 4. JSON 응답 형태로 변환
#         recommendations = []
#         for movie_id, score in top_recommendations:
#             movie_info = get_movie_info(movie_id)  # CSV에서 영화 정보 조회
#             recommendations.append(MovieRecommendation(
#                 movie_id=movie_id,
#                 title=movie_info['title'],
#                 score=round(score, 3),
#                 genres=movie_info['genres'].split(','),
#                 main_actors=movie_info['actors'].split(','),
#                 directors=movie_info['directors'].split(',')
#             ))
        
#         return RecommendationResponse(
#             user_id=user_id,
#             recommendations=recommendations,
#             total_count=len(recommendations),
#             generated_at=datetime.now().isoformat()
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))