import logging
import pandas as pd
from typing import List, Optional, Any
from fastapi import HTTPException
from .state import fr_app_state
from pydantic import BaseModel, Field # PopularContentResponse를 여기서도 사용하기 위해 임포트
from .config import (
    LIKE_WEIGHT, DISLIKE_WEIGHT, STAR_RATING_WEIGHTS,
    MIN_INTERACTIONS_FOR_POPULARITY,
    MIN_INTERACTIONS_PER_PERSONA_POPULAR
)

logger = logging.getLogger(__name__)

class PopularContentResponse(BaseModel):
    content_id: int = Field(..., alias="contentId")
    title: str
    type: str
    poster_path: Optional[str] = None
    genres: List[str] = []
    platforms: List[str] = [] # <--- 이 줄 추가!
    popularity_score: Optional[float] = None

# ===============================================
# 연령대별 인기 콘텐츠 계산 로직
# ===============================================
def calculate_age_group_popularity(
    age_group_min: int,
    age_group_max: int,
    top_n: int = 10
) -> List[PopularContentResponse]:
    """
    특정 연령대 그룹에 속하는 사용자들의 반응을 기반으로 인기 콘텐츠를 계산하고,
    PopularContentResponse 모델 리스트로 반환합니다.
    """
    logger.info(f"연령대 {age_group_min}~{age_group_max} 인기 콘텐츠 계산 시작.")

    if fr_app_state.users_df is None or fr_app_state.reactions_df is None or \
       fr_app_state.reviews_df is None or fr_app_state.contents_df is None:
        raise HTTPException(status_code=503, detail="필요한 데이터가 아직 로드되지 않았습니다.")

    # 1. 해당 연령대 사용자 필터링
    target_users_df = fr_app_state.users_df[
        (fr_app_state.users_df['age'] >= age_group_min) & 
        (fr_app_state.users_df['age'] <= age_group_max)
    ]
    if target_users_df.empty:
        logger.warning(f"연령대 {age_group_min}~{age_group_max}에 해당하는 사용자가 없습니다.")
        return []

    target_user_ids = target_users_df['user_id'].tolist()

    # 2. 해당 사용자들의 반응 데이터 필터링
    user_reactions = fr_app_state.reactions_df[
        fr_app_state.reactions_df['user_id'].isin(target_user_ids)
    ].copy()
    user_reviews = fr_app_state.reviews_df[
        fr_app_state.reviews_df['user_id'].isin(target_user_ids)
    ].copy()

    if user_reactions.empty and user_reviews.empty:
        logger.warning(f"연령대 {age_group_min}~{age_group_max} 사용자의 반응 데이터가 없습니다.")
        return []

    # 3. 반응 점수화 및 통합
    user_reactions['score'] = user_reactions['reaction'].apply(
        lambda x: LIKE_WEIGHT if x == 'LIKE' else (DISLIKE_WEIGHT if x == 'DISLIKE' else 0)
    )
    user_reactions_processed = user_reactions[['content_id', 'type', 'score']]

    user_reviews_processed = user_reviews[['content_id', 'type', 'score']].copy()
    user_reviews_processed['score'] = user_reviews_processed['score'].apply(
        lambda x: STAR_RATING_WEIGHTS.get(x, 0.0)
    )

    all_interactions = pd.concat([user_reactions_processed, user_reviews_processed], ignore_index=True)
    
    # 4. 콘텐츠별 인기 점수 집계
    content_popularity = all_interactions.groupby(['content_id', 'type']).agg(
        total_score=('score', 'sum'),
        interaction_count=('score', 'count')
    ).reset_index()

    content_popularity = content_popularity[content_popularity['interaction_count'] >= MIN_INTERACTIONS_FOR_POPULARITY]
    content_popularity['popularity_score'] = content_popularity['total_score'] 

    if content_popularity.empty:
        logger.warning(f"연령대 {age_group_min}~{age_group_max}의 충분한 반응을 가진 인기 콘텐츠가 없습니다.")
        return []

    # 5. 상위 N개 콘텐츠 선택
    top_contents = content_popularity.nlargest(top_n, 'popularity_score')

    # 6. 콘텐츠 정보 병합 및 응답 형식으로 변환
    merged_data = pd.merge(
        top_contents,
        fr_app_state.contents_df,
        on=['content_id', 'type'],
        how='left'
    )
    
    recommendations = []
    for _, row in merged_data.iterrows():
        recommendations.append(PopularContentResponse(
            contentId=row['content_id'],
            title=row['title'],
            type=row['type'],
            poster_path=row['poster_path'],
            genres=row['genres'],
            platforms=row['platforms'], # <--- 이 줄 추가!
            popularity_score=row['popularity_score']
        ))
    
    logger.info(f"연령대 {age_group_min}~{age_group_max} 인기 콘텐츠 계산 완료: {len(recommendations)}개.")
    return recommendations

# ===============================================
# 페르소나별 인기 콘텐츠 계산 로직
# ===============================================
def calculate_popular_by_persona(
    persona_id: int, 
    top_n: int = 10
) -> List[PopularContentResponse]:
    """
    특정 페르소나에 속하는 사용자들의 반응을 기반으로 인기 콘텐츠를 계산하고,
    PopularContentResponse 모델 리스트로 반환합니다.
    """
    logger.info(f"페르소나 ID {persona_id} 인기 콘텐츠 계산 시작.")

    if fr_app_state.all_user_personas_df is None or fr_app_state.reactions_df is None or \
       fr_app_state.reviews_df is None or fr_app_state.contents_df is None:
        raise HTTPException(status_code=503, detail="필요한 데이터가 아직 로드되지 않았습니다.")

    persona_name = fr_app_state.persona_id_to_name_map.get(persona_id)
    if persona_name is None:
        raise HTTPException(status_code=404, detail=f"페르소나 ID {persona_id}를 찾을 수 없습니다.")

    # 1. 해당 페르소나에 속하는 사용자 ID 찾기
    main_persona_ids_per_user = fr_app_state.all_user_personas_df.loc[
        fr_app_state.all_user_personas_df.groupby('user_id')['score'].idxmax()
    ]
    target_user_ids_df = main_persona_ids_per_user[main_persona_ids_per_user['persona_id'] == persona_id]
    target_user_ids = target_user_ids_df['user_id'].tolist()

    if not target_user_ids:
        logger.warning(f"페르소나 '{persona_name}'에 해당하는 사용자가 없습니다.")
        return []

    # 2. 해당 사용자들의 반응 데이터 필터링
    user_reactions = fr_app_state.reactions_df[
        fr_app_state.reactions_df['user_id'].isin(target_user_ids)
    ].copy()
    user_reviews = fr_app_state.reviews_df[
        fr_app_state.reviews_df['user_id'].isin(target_user_ids)
    ].copy()

    if user_reactions.empty and user_reviews.empty:
        logger.warning(f"페르소나 '{persona_name}' 사용자의 반응 데이터가 없습니다.")
        return []

    # 3. 반응 점수화 및 통합
    user_reactions['score'] = user_reactions['reaction'].apply(
        lambda x: LIKE_WEIGHT if x == 'LIKE' else (DISLIKE_WEIGHT if x == 'DISLIKE' else 0)
    )
    user_reactions_processed = user_reactions[['content_id', 'type', 'score']]

    user_reviews_processed = user_reviews[['content_id', 'type', 'score']].copy()
    user_reviews_processed['score'] = user_reviews_processed['score'].apply(
        lambda x: STAR_RATING_WEIGHTS.get(x, 0.0)
    )

    all_interactions = pd.concat([user_reactions_processed, user_reviews_processed], ignore_index=True)
    
    # 4. 콘텐츠별 인기 점수 집계
    content_popularity = all_interactions.groupby(['content_id', 'type']).agg(
        total_score=('score', 'sum'),
        interaction_count=('score', 'count')
    ).reset_index()

    content_popularity = content_popularity[content_popularity['interaction_count'] >= MIN_INTERACTIONS_PER_PERSONA_POPULAR]
    content_popularity['popularity_score'] = content_popularity['total_score'] 

    if content_popularity.empty:
        logger.warning(f"페르소나 '{persona_name}'의 충분한 반응을 가진 인기 콘텐츠가 없습니다.")
        return []

    # 5. 상위 N개 콘텐츠 선택
    top_contents = content_popularity.nlargest(top_n, 'popularity_score')

    # 6. 콘텐츠 정보 병합 및 응답 형식으로 변환
    merged_data = pd.merge(
        top_contents,
        fr_app_state.contents_df,
        on=['content_id', 'type'],
        how='left'
    )

    recommendations = []
    for _, row in merged_data.iterrows():
        recommendations.append(PopularContentResponse(
            contentId=row['content_id'],
            title=row['title'],
            type=row['type'],
            poster_path=row['poster_path'],
            genres=row['genres'],
            platforms=row['platforms'], # <--- 이 줄 추가!
            popularity_score=row['popularity_score']
        ))
    
    logger.info(f"페르소나 ID {persona_id} 인기 콘텐츠 계산 완료: {len(recommendations)}개.")
    return recommendations