import logging
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import pandas as pd
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field 

from database import get_db

from . import filtering_generator
from .state import fr_app_state 
from .config import RECOMMENDATION_COUNT

logger = logging.getLogger(__name__)

filtering_recommender_router = APIRouter()

class PopularContentResponse(BaseModel):
    content_id: int = Field(..., alias="contentId", description="콘텐츠 ID")
    title: str = Field(..., description="콘텐츠 제목")
    type: str = Field(..., description="콘텐츠 유형 (movie/tv)")
    poster_path: Optional[str] = Field(None, description="포스터 이미지 경로")
    genres: List[str] = Field([], description="장르 목록")
    platforms: List[str] = Field([], description="제공되는 플랫폼 목록")
    popularity_score: Optional[float] = Field(None, description="계산된 인기 점수")


class PopularContentListResponse(BaseModel):
    message: str
    recommendations: List[PopularContentResponse]
    group_info: Optional[str] = None
    total_count: int


@filtering_recommender_router.get(
    "/popular-by-age-group",
    response_model=PopularContentListResponse,
    summary="연령대별 인기 콘텐츠 조회"
)
async def get_popular_by_age_group(
    age_group_min: int = Query(0, ge=0, description="연령대 최소 값 (예: 10, 20)"),
    age_group_max: int = Query(100, le=100, description="연령대 최대 값 (예: 19, 29)"),
    limit: int = Query(RECOMMENDATION_COUNT, gt=0, le=100, description="반환할 콘텐츠 수")
):
    """
    지정된 연령대 범위의 사용자들에게 인기 있는 콘텐츠를 반환합니다.
    """
    logger.info(f"연령대 {age_group_min}~{age_group_max}에 대한 인기 콘텐츠 요청 접수.")

    try:
        raw_recommendations = filtering_generator.calculate_age_group_popularity(
            age_group_min=age_group_min,
            age_group_max=age_group_max,
            top_n=limit
        )
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

@filtering_recommender_router.get(
    "/personas/popular-by-persona/{persona_id}",
    response_model=PopularContentListResponse,
    summary="특정 페르소나별 인기 콘텐츠 조회"
)
async def get_popular_by_persona_id(
    persona_id: int, 
    limit: int = Query(RECOMMENDATION_COUNT, gt=0, le=100, description="반환할 콘텐츠 수")
):
    """
    지정된 페르소나 ID에 속하는 사용자들에게 인기 있는 콘텐츠를 반환합니다.
    """
    logger.info(f"페르소나 ID '{persona_id}'에 대한 인기 콘텐츠 요청 접수.")

    persona_name = fr_app_state.persona_id_to_name_map.get(persona_id)
    if persona_name is None:
        raise HTTPException(status_code=404, detail=f"페르소나 ID '{persona_id}'를 찾을 수 없습니다.")

    try:
        raw_recommendations = filtering_generator.calculate_popular_by_persona(
            persona_id=persona_id,
            top_n=limit
        )
        
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
