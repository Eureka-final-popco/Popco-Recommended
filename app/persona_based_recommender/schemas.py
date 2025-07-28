from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum

# 2.1. POST /recommend/persona/onboard 요청 본문 (InitialFeedbackRequest)
class ContentFeedbackItem(BaseModel):
    content_id: int
    content_type: str # "movie", "tv" 등

class InitialFeedbackRequest(BaseModel):
    user_id: int
    feedback_items: List[ContentFeedbackItem] = Field(..., min_items=3, description="사용자가 선호하는 콘텐츠 목록 (최소 3개).")
    reaction_type: str = Field(..., pattern="^좋아요$", description="온보딩 시에는 반드시 '좋아요'여야 합니다.")
    initial_answers: Dict[str, str]

# 2.1. POST /recommend/persona/onboard 및 2.2. POST /recommend/persona/feedback 응답 (RecommendationResponse)
class RecommendedContent(BaseModel):
    contentId: int
    title: Optional[str] = None
    genres: List[str] = []
    type: Optional[str] = None
    poster_path: Optional[str] = None  # 새로 추가된 필드
    predicted_rating: Optional[float] = None
    persona_genre_match: Optional[bool] = None

class RecommendationResponse(BaseModel):
    message: str
    recommendations: List[RecommendedContent]
    main_persona: str
    sub_persona: Optional[str] = None
    all_personas_scores: Dict[str, float]

# 2.2. POST /recommend/persona/feedback 요청 본문 (FeedbackRequest)
class FeedbackRequest(BaseModel):
    user_id: int
    content_id: int
    content_type: str
    reaction_type: str = Field(..., description="피드백 유형 ('좋아요', '싫어요', '평점').")
    score: Optional[float] = Field(None, ge=0.0, le=5.0, description="reaction_type이 '평점'일 경우 필수 (0.0 ~ 5.0).")

    # 평점일 경우 score 필드가 필수임을 검증 (Pydantic v2에서는 model_validator 사용)
    # 현재는 FastAPI가 기본적으로 유효성 검사를 처리하므로 주석 처리
    # @root_validator(pre=True)
    # def check_score_for_rating(cls, values):
    #     if values.get('reaction_type') == '평점' and values.get('score') is None:
    #         raise ValueError("reaction_type이 '평점'일 경우 score 필드는 필수입니다.")
    #     return values

# 2.3. GET /recommend/persona/users/{user_id}/recommendations 쿼리 파라미터 (RecommendationRequest)
# 이 모델은 FastAPI의 Query 파라미터로 직접 매핑되므로, 이 스키마 자체는 직접적으로 요청 본문 모델로 사용되지 않습니다.
# 내부 로직에서 활용될 수 있도록 유지합니다.
class RecommendationRequest(BaseModel):
    num_recommendations: int = 10 # 명세에 없지만 기존 로직에 있으므로 기본값 유지
    content_type_filter: Optional[str] = None # 'movie', 'tv' 또는 None

# 2.4. GET /recommend/persona/persona/counts 응답 (PersonaCountsResponse)
class PersonaCountsResponse(BaseModel):
    message: str
    persona_user_counts: Dict[str, int]
    total_users_with_persona: int