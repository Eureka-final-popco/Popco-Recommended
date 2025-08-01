import logging
from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
import pandas as pd
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field # BaseModel, Field 임포트 추가

# DB 모델 임포트 경로 수정 (프로젝트 루트 기준으로)
from database import get_db

# 내부 모듈 임포트 경로 수정
from . import filtering_generator
from .state import fr_app_state # 변경된 state 객체 이름
from .config import RECOMMENDATION_COUNT # config 파일에서 필요한 상수 임포트

# schemas 임포트: 여기서 RecommendedContent는 persona_based_recommender의 것을 사용하고
# PopularContentResponse는 이 파일에서 새로 정의합니다.
# from persona_based_recommender.schemas import RecommendedContent # 이제는 필요 없을 수 있음 (PopularContentResponse로 대체)

logger = logging.getLogger(__name__)

# 라우터 이름 및 프리픽스
filtering_recommender_router = APIRouter(prefix="/recommends/filters")

# ==== 응답 스키마 정의 (filtering 모듈 전용) ====
class PopularContentResponse(BaseModel):
    content_id: int = Field(..., alias="contentId", description="콘텐츠 ID")
    title: str = Field(..., description="콘텐츠 제목")
    type: str = Field(..., description="콘텐츠 유형 (movie/tv)")
    poster_path: Optional[str] = Field(None, description="포스터 이미지 경로")
    genres: List[str] = Field([], description="장르 목록")
    platforms: List[str] = Field([], description="제공되는 플랫폼 목록") # ===> platforms 필드 추가
    popularity_score: Optional[float] = Field(None, description="계산된 인기 점수")


class PopularContentListResponse(BaseModel):
    message: str
    recommendations: List[PopularContentResponse]
    group_info: Optional[str] = None
    total_count: int



# ===============================================
# 연령대별 인기 콘텐츠 API
# ===============================================
@filtering_recommender_router.get(
    "/popular-by-age-group",
    response_model=PopularContentListResponse,
    summary="연령대별 인기 콘텐츠 조회"
)
async def get_popular_by_age_group(
    age_group_min: int = Query(0, ge=0, description="연령대 최소 값 (예: 10, 20)"),
    age_group_max: int = Query(100, le=100, description="연령대 최대 값 (예: 19, 29)"),
    limit: int = Query(RECOMMENDATION_COUNT, gt=0, le=50, description="반환할 콘텐츠 수")
):
    """
    지정된 연령대 범위의 사용자들에게 인기 있는 콘텐츠를 반환합니다.
    """
    logger.info(f"연령대 {age_group_min}~{age_group_max}에 대한 인기 콘텐츠 요청 접수.")

    try:
        raw_recommendations = filtering_generator.calculate_age_group_popularity( # 변수명 변경
            age_group_min=age_group_min,
            age_group_max=age_group_max,
            top_n=limit
        )
        # Pydantic 모델 객체를 딕셔너리로 변환
        recommendations = [rec.model_dump(by_alias=True) for rec in raw_recommendations]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"연령대별 인기 콘텐츠 계산 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"연령대별 인기 콘텐츠 생성 중 오류 발생: {str(e)}")

    if not recommendations:
        raise HTTPException(status_code=404, detail=f"연령대 {age_group_min}-{age_group_max}를 위한 인기 콘텐츠를 찾을 수 없습니다.")

    return PopularContentListResponse(
        message=f"연령대 {age_group_min}-{age_group_max}세를 위한 인기 콘텐츠입니다.",
        recommendations=recommendations,
        group_info=f"{age_group_min}-{age_group_max}세",
        total_count=len(recommendations)
    )

# ===============================================
# 페르소나별 인기 콘텐츠 API
# ===============================================
@filtering_recommender_router.get(
    "/personas/popular-by-persona/{persona_id}",
    response_model=PopularContentListResponse,
    summary="특정 페르소나별 인기 콘텐츠 조회"
)
async def get_popular_by_persona_id(
    persona_id: int, # Query가 아니라 Path 파라미터로 변경
    limit: int = Query(RECOMMENDATION_COUNT, gt=0, le=50, description="반환할 콘텐츠 수")
):
    """
    지정된 페르소나 ID에 속하는 사용자들에게 인기 있는 콘텐츠를 반환합니다.
    """
    logger.info(f"페르소나 ID '{persona_id}'에 대한 인기 콘텐츠 요청 접수.")

    persona_name = fr_app_state.persona_id_to_name_map.get(persona_id)
    if persona_name is None:
        raise HTTPException(status_code=404, detail=f"페르소나 ID '{persona_id}'를 찾을 수 없습니다.")

    try:
        raw_recommendations = filtering_generator.calculate_popular_by_persona( # 변수명 변경
            persona_id=persona_id,
            top_n=limit
        )
        # Pydantic 모델 객체를 딕셔너리로 변환
        recommendations = [rec.model_dump(by_alias=True) for rec in raw_recommendations]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"페르소나별 인기 콘텐츠 계산 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"페르소나별 인기 콘텐츠 생성 중 오류 발생: {str(e)}")

    if not recommendations:
        raise HTTPException(status_code=404, detail=f"페르소나 '{persona_name}'를 위한 인기 콘텐츠를 찾을 수 없습니다.")

    return PopularContentListResponse(
        message=f"'{persona_name}' 페르소나를 위한 인기 콘텐츠입니다.",
        recommendations=recommendations,
        group_info=persona_name,
        total_count=len(recommendations)
    )
