from pydantic import BaseModel
from typing import List, Optional

class RecommendRequest(BaseModel):
    type: str = "all"
    user_id: Optional[int] = None

class ContentRecommendationListResponse(BaseModel):
    content_id: int
    content_type: Optional[str] = None
    title: str
    total_similarity: float
    poster_path: Optional[str] = None

    
# 응답용 스키마 수정
class ContentResponse(BaseModel):
    content_id: int
    content_type: str
    title: str
    poster_path: Optional[str] = None
    user_reaction: Optional[str] = None

class RecommendationListResponse(BaseModel):
    recommendations: List[ContentResponse]