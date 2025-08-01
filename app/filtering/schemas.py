from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

class RecommendedContent(BaseModel):
    contentId: int
    title: Optional[str] = None
    genres: List[str] = []
    type: Optional[str] = None
    poster_path: Optional[str] = None
    predicted_rating: Optional[float] = None
    persona_genre_match: Optional[bool] = None

class RecommendationResponse(BaseModel):
    message: str
    recommendations: List[RecommendedContent]
    main_persona: str # 이 필드는 '나에게 맞는 콘텐츠'에서만 유의미할 수 있습니다.
    sub_persona: Optional[str] = None
    all_personas_scores: Dict[str, float] # 이 필드는 '나에게 맞는 콘텐츠'에서만 유의미할 수 있습니다.
