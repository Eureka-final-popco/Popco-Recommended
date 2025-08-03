from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ContentFeedbackItem(BaseModel):
    content_id: int
    content_type: str

class InitialFeedbackRequest(BaseModel):
    user_id: int
    feedback_items: List[ContentFeedbackItem] = Field(..., min_items=3, description="사용자가 선호하는 콘텐츠 목록 (최소 3개).")
    reaction_type: str = Field(..., pattern="^좋아요$", description="온보딩 시에는 반드시 '좋아요'여야 합니다.")
    initial_answers: Dict[str, str]

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
    main_persona: str
    sub_persona: Optional[str] = None
    all_personas_scores: Dict[str, float]

class FeedbackRequest(BaseModel):
    user_id: int
    content_id: int
    content_type: str
    reaction_type: str = Field(..., description="피드백 유형 ('좋아요', '싫어요', '평점').")
    score: Optional[float] = Field(None, ge=0.0, le=5.0, description="reaction_type이 '평점'일 경우 필수 (0.0 ~ 5.0).")

class RecommendationRequest(BaseModel):
    num_recommendations: int = 10 
    content_type_filter: Optional[str] = None

class PersonaCountsResponse(BaseModel):
    message: str
    persona_user_counts: Dict[str, int]
    total_users_with_persona: int