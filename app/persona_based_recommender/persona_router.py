import logging
from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func, text
from datetime import datetime

# DB 모델 임포트
from app.database import get_db, SQLALCHEMY_DATABASE_URL
from app.models import ContentReaction, UserPersona, Review, ReactionType, User

# 내부 모듈 임포트
from . import persona_generator
from .state import pbr_app_state
from .config import (
    RECOMMENDATION_COUNT,
    MIN_POSITIVE_RATINGS_FOR_GROWTH,
    QA_INITIAL_RECOMMENDATION_THRESHOLD
)
from app.persona_based_recommender.persona_generator import (
    calculate_user_content_matrix_sparse, 
    calculate_user_similarity,      
)
from .schemas import InitialFeedbackRequest, FeedbackRequest, RecommendationResponse, RecommendedContent, PersonaCountsResponse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

persona_recommender_router = APIRouter(prefix="/recommend/persona")


def calculate_and_store_user_personas(
    db: Session,
    user_id: int, # 단일 사용자 ID를 받도록 변경
    calculated_persona_scores: Dict[str, float] # 계산된 페르소나 점수 딕셔너리를 받도록 변경
):
    """
    주어진 사용자 ID에 대해 계산된 페르소나 점수를 DB 및 전역 상태를 업데이트합니다.
    """
    if pbr_app_state.contents_df is None or pbr_app_state.reactions_df is None:
        logger.error("데이터 로드되지 않음. 페르소나 계산 불가.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="추천 시스템 데이터가 로드되지 않았습니다.")

    logger.info(f"사용자 [{user_id}]의 페르소나 점수 DB 및 상태 저장 시작.")

    persona_name_to_id_map = pbr_app_state.persona_name_to_id_map
    if persona_name_to_id_map is None or not persona_name_to_id_map:
        logger.error("페르소나 이름-ID 매핑이 초기화되지 않았습니다. 페르소나 점수 저장 불가.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="페르소나 매핑 데이터가 없습니다.")

    new_user_persona_rows = []
    for persona_name, score in calculated_persona_scores.items():
        persona_id = persona_name_to_id_map.get(persona_name)
        if persona_id is not None:
            new_user_persona_rows.append({
                'user_id': user_id,
                'persona_id': persona_id,
                'score': score,
                'created_at': datetime.now(), # pd.Timestamp.now() 대신 datetime.now() 사용
                'updated_at': datetime.now()
            })
    
    if not new_user_persona_rows:
        logger.warning(f"사용자 {user_id}에 대해 저장할 페르소나 점수가 없습니다.")
        return

    new_persona_scores_df = pd.DataFrame(new_user_persona_rows)

    # pbr_app_state.all_user_personas_df 업데이트
    if pbr_app_state.all_user_personas_df is not None and not pbr_app_state.all_user_personas_df.empty:
        # 기존 사용자 페르소나 제거 (업데이트를 위해)
        pbr_app_state.all_user_personas_df = pbr_app_state.all_user_personas_df[
            pbr_app_state.all_user_personas_df['user_id'] != user_id
        ]
        pbr_app_state.all_user_personas_df = pd.concat([pbr_app_state.all_user_personas_df, new_persona_scores_df], ignore_index=True)
    else:
        pbr_app_state.all_user_personas_df = new_persona_scores_df

    # DB에 저장 (upsert 로직)
    for _, row in new_persona_scores_df.iterrows():
        user_persona = db.query(UserPersona).filter_by(
            user_id=int(row['user_id']),
            persona_id=int(row['persona_id'])
        ).first()

        if user_persona:
            user_persona.score = float(row['score'])
            user_persona.updated_at = func.now()
        else:
            user_persona = UserPersona(
                user_id=int(row['user_id']),
                persona_id=int(row['persona_id']),
                score=float(row['score']),
                created_at=func.now(),
                updated_at=func.now()
            )
            db.add(user_persona)
    db.commit()
    logger.info(f"DB/State: 사용자 [{user_id}]의 페르소나 점수 {len(new_persona_scores_df)}개 저장/업데이트 완료.")


@persona_recommender_router.post("/onboard", response_model=RecommendationResponse, status_code=status.HTTP_200_OK)
async def onboard_user(
    request: InitialFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    신규 사용자 온보딩: 초기 피드백(선호 콘텐츠 좋아요)과 QA 답변을 기반으로 페르소나를 결정하고 추천을 제공합니다.
    QA 답변은 DB에 저장되지 않고, 페르소나 계산에만 사용됩니다.
    """
    user_id = request.user_id
    logger.info(f"사용자 {user_id} 온보딩 시작.")

    if pbr_app_state.contents_df.empty or pbr_app_state.reactions_df is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="추천 시스템 데이터가 아직 로드되지 않았거나 비어 있습니다.")

    db_user = db.query(User).filter(User.user_id == user_id).first()
    if not db_user:
        try:
            new_user = User(
                user_id=user_id,
                email=f"user_{user_id}@example.com", # 임시 이메일
                name=f"User {user_id}",               # 임시 이름
                password="hashed_password",          # 실제 환경에서는 해싱된 비밀번호 사용
                is_active=True
            )
            db.add(new_user)
            db.flush()
            logger.info(f"DB: 사용자 {user_id}를 users 테이블에 새로 생성했습니다.")
        except Exception as e:
            db.rollback()
            logger.error(f"사용자 {user_id} 생성 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="사용자 생성 실패.")


    # 2. 초기 피드백 (ContentReaction) DB 저장
    new_reactions = []
    for item in request.feedback_items:
        # contentId_orig 대신 content_id 사용
        content_exists = pbr_app_state.contents_df[
            (pbr_app_state.contents_df['content_id'] == item.content_id) & # 수정
            (pbr_app_state.contents_df['type'] == item.content_type)
        ].any().any()

        if not content_exists:
            logger.warning(f"콘텐츠 ID {item.content_id}, 타입 {item.content_type}은 존재하지 않는 콘텐츠입니다. 건너뜁니다.")
            continue

        new_reactions.append(
            ContentReaction(
                user_id=user_id,
                content_id=item.content_id,
                type=item.content_type,
                reaction=ReactionType.LIKE
            )
        )
    if new_reactions:
        db.add_all(new_reactions)
        logger.info(f"DB: 사용자 {user_id}의 초기 콘텐츠 반응 {len(new_reactions)}개 저장 대기 중.")
        
        new_reactions_df_data = []
        for r in new_reactions:
            new_reactions_df_data.append({
                'user_id': r.user_id,
                'content_id': r.content_id, # contentId_orig 대신 content_id 사용
                'type': r.type,
                'reaction': r.reaction.value,
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            })
        new_reactions_df = pd.DataFrame(new_reactions_df_data)

        if pbr_app_state.reactions_df is not None and not pbr_app_state.reactions_df.empty:
            # 기존 reactions_df에 추가하기 전에 중복을 제거
            # contentId_orig 대신 content_id 사용
            existing_reactions_keys = pbr_app_state.reactions_df[
                (pbr_app_state.reactions_df['user_id'] == user_id) &
                (pbr_app_state.reactions_df['reaction'] == ReactionType.LIKE.value)
            ].set_index(['user_id', 'content_id', 'type']).index # 수정
            
            new_reactions_df_filtered = new_reactions_df[
                ~new_reactions_df.set_index(['user_id', 'content_id', 'type']).index.isin( # 수정
                    existing_reactions_keys
                )
            ]
            pbr_app_state.reactions_df = pd.concat([pbr_app_state.reactions_df, new_reactions_df_filtered], ignore_index=True)
            logger.info(f"pbr_app_state.reactions_df에 새로운 반응 {len(new_reactions_df_filtered)}개 추가됨.")
        else:
            pbr_app_state.reactions_df = new_reactions_df
            logger.info(f"pbr_app_state.reactions_df가 초기화되고 {len(new_reactions_df)}개 반응이 추가됨.")
    else:
        logger.warning(f"사용자 {user_id}에 대한 유효한 초기 콘텐츠 반응이 없습니다.")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="유효한 초기 피드백 콘텐츠가 제공되지 않았습니다.")

    # 3. 페르소나 계산 및 DB/pbr_app_state 업데이트
    unified_feedback = persona_generator.generate_unified_user_feedback(user_id, db) # db 인자 전달

    main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores = \
        persona_generator.get_hybrid_persona(user_id, unified_feedback, db) # db 인자 전달

    # calculate_and_store_user_personas 호출 (db.commit()은 나중에)
    calculate_and_store_user_personas(db, user_id=user_id, calculated_persona_scores=all_personas_scores)
    
    try:
        db.commit()
        logger.info(f"DB: 사용자 {user_id} 온보딩 관련 모든 변경사항 커밋 완료.")

        # --- 추가된 부분: DB 커밋 후 pbr_app_state의 user_personas 데이터 갱신 ---
        logger.info(f"온보딩 후 메모리 내 pbr_app_state.all_user_personas_df 및 user_persona_scores_map 갱신 시도.")
        result = db.execute(text("SELECT * FROM user_personas"))
        all_user_personas_df_updated = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.all_user_personas_df = all_user_personas_df_updated
        
        pbr_app_state.user_persona_scores_map = {}
        for _, row in pbr_app_state.all_user_personas_df.iterrows():
            user_id_from_df = row['user_id']
            persona_id = row['persona_id']
            score = row['score']
            if user_id_from_df not in pbr_app_state.user_persona_scores_map:
                pbr_app_state.user_persona_scores_map[user_id_from_df] = {}
            pbr_app_state.user_persona_scores_map[user_id_from_df][persona_id] = score
        logger.info(f"pbr_app_state.all_user_personas_df 및 user_persona_scores_map 갱신 완료.")
        # --- 추가된 부분 끝 ---

    except Exception as e:
        db.rollback()
        logger.error(f"사용자 {user_id} 온보딩 중 DB 커밋 또는 메모리 갱신 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="DB 저장 중 오류 발생.")

    # 4. 추천 생성 및 반환
    recommendations: List[RecommendedContent] = []

    if main_persona_id is not None and main_persona_name != "Undetermined":
        logger.info(f"사용자 {user_id}의 메인 페르소나({main_persona_name})를 기반으로 추천을 생성합니다.")
        # update_and_get_recommendations 함수는 내부적으로 update_user_persona_scores를 호출하므로,
        # onboard_user 에서는 이미 DB 업데이트와 state 갱신이 이루어졌기에,
        # 단순히 추천 결과를 가져오는 함수로 대체하거나, update_and_get_recommendations 내부 로직을 점검해야 합니다.
        # 여기서는 persona_generator.py의 update_and_get_recommendations가 적절히 작동한다고 가정하고 db 인자를 전달합니다.
        recommendations = persona_generator.update_and_get_recommendations( # 이 함수가 너무 많은 일을 한다면 분리 고려
            user_id=user_id,
            db=db,
            num_recommendations=RECOMMENDATION_COUNT,
            content_type_filter=None
        )
    else:
        logger.info(f"사용자 {user_id}의 페르소나 기반 추천 생성 불가 (페르소나 미결정), 일반 인기 추천으로 대체.")
        recommendations = persona_generator._get_general_popular_recommendations(user_id, num_recommendations=RECOMMENDATION_COUNT, db=db) # db 인자 전달

    logger.info(f"사용자 {user_id} 온보딩 완료. {len(recommendations)}개 추천.")

    # all_personas_scores의 키를 페르소나 이름으로 변환하고 점수를 1-100으로 정규화
    normalized_and_named_scores = {}
    if all_personas_scores:
        total_score = sum(all_personas_scores.values())
        
        # pbr_app_state.persona_df에서 페르소나 ID와 이름 매핑
        persona_id_to_name = pbr_app_state.persona_df.set_index('persona_id')['name'].to_dict() # 'persona_name'이 아니라 'name' 컬럼 사용

        for persona_id, score in all_personas_scores.items():
            persona_name = persona_id_to_name.get(persona_id, f"{persona_id}")
            if total_score > 0:
                normalized_score = (score / total_score) * 100
            else:
                normalized_score = 0.0
            normalized_and_named_scores[persona_name] = round(normalized_score, 2)

    return RecommendationResponse(
        message="온보딩 완료 및 초기 추천이 제공되었습니다.",
        recommendations=recommendations,
        main_persona=main_persona_name,
        sub_persona=sub_persona_name,
        all_personas_scores=normalized_and_named_scores
    )


@persona_recommender_router.post("/feedback", response_model=RecommendationResponse, status_code=status.HTTP_200_OK)
async def submit_feedback(
    request: FeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    사용자 피드백(좋아요/싫어요/평점)을 받아 페르소나를 업데이트하고 새로운 추천을 제공합니다.
    """
    user_id = request.user_id
    content_id = request.content_id
    content_type = request.content_type
    reaction_type_str = request.reaction_type # 문자열로 받은 reaction_type
    score = request.score

    logger.info(f"사용자 {user_id}의 피드백 접수: 콘텐츠 {content_id}, 타입 {content_type}, 반응 {reaction_type_str}, 점수 {score}")

    if pbr_app_state.contents_df.empty:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="추천 시스템 데이터가 아직 로드되지 않았거나 비어 있습니다.")

    content_exists = pbr_app_state.contents_df[
        (pbr_app_state.contents_df['content_id'] == content_id) &
        (pbr_app_state.contents_df['type'] == content_type)
    ].any().any()

    if not content_exists:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"콘텐츠 ID {content_id}, 타입 {content_type}을 찾을 수 없습니다.")

    try:
        # 1. 사용자 피드백을 DB에 저장하고 pbr_app_state 갱신 (반응 또는 평점)
        if reaction_type_str in ['좋아요', '싫어요']:
            db_reaction_enum = None # <-- ReactionType enum 멤버를 저장할 변수
            if reaction_type_str == '좋아요':
                db_reaction_enum = ReactionType.LIKE
            elif reaction_type_str == '싫어요':
                db_reaction_enum = ReactionType.DISLIKE
            
            if db_reaction_enum is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="유효하지 않은 반응 유형입니다.")

            # ContentReaction 모델에 'reaction' 컬럼을 사용하도록 수정
            existing_reaction = db.query(ContentReaction).filter_by(
                user_id=user_id, content_id=content_id, type=content_type
            ).first()
            if existing_reaction:
                existing_reaction.reaction = db_reaction_enum # <-- 중요: SQLEnum 필드에는 Enum 멤버를 직접 할당
                existing_reaction.updated_at = func.now()
                db.add(existing_reaction) # 변경사항을 세션에 다시 추가
            else:
                db.add(ContentReaction(user_id=user_id, content_id=content_id, type=content_type, reaction=db_reaction_enum, created_at=datetime.utcnow())) # <-- 중요: SQLEnum 필드에는 Enum 멤버를 직접 할당
            
            # pbr_app_state.reactions_df 업데이트 로직 (DataFrame 컬럼에는 보통 문자열 값을 저장)
            if pbr_app_state.reactions_df is not None:
                mask = (pbr_app_state.reactions_df['user_id'] == user_id) & \
                       (pbr_app_state.reactions_df['content_id'] == content_id) & \
                       (pbr_app_state.reactions_df['type'] == content_type)
                if mask.any():
                    pbr_app_state.reactions_df.loc[mask, 'reaction'] = db_reaction_enum.value # <-- DataFrame에는 .value 사용
                    pbr_app_state.reactions_df.loc[mask, 'updated_at'] = pd.Timestamp.now()
                else:
                    new_row_df = pd.DataFrame([{
                        'user_id': user_id, 'content_id': content_id, 'type': content_type,
                        'reaction': db_reaction_enum.value, # <-- DataFrame에는 .value 사용
                        'created_at': pd.Timestamp.now(), 'updated_at': pd.Timestamp.now()
                    }])
                    pbr_app_state.reactions_df = pd.concat([pbr_app_state.reactions_df, new_row_df], ignore_index=True)
            else:
                pbr_app_state.reactions_df = pd.DataFrame([{
                    'user_id': user_id, 'content_id': content_id, 'type': content_type,
                    'reaction': db_reaction_enum.value, # <-- DataFrame에는 .value 사용
                    'created_at': pd.Timestamp.now(), 'updated_at': pd.Timestamp.now()
                }])
            logger.info(f"DB/State: 사용자 {user_id}의 콘텐츠 반응({reaction_type_str}) 저장/업데이트 준비 완료.")


        elif reaction_type_str == '평점':
            if score is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="평점 반응에는 'score'가 필수입니다.")
            
            existing_review = db.query(Review).filter_by(
                user_id=user_id, content_id=content_id, type=content_type
            ).first()
            if existing_review:
                existing_review.score = score
                existing_review.updated_at = func.now()
                db.add(existing_review) # 변경사항을 세션에 다시 추가
            else:
                db.add(Review(
                    user_id=user_id,
                    content_id=content_id,
                    type=content_type,
                    score=score,
                    created_at=datetime.utcnow() # created_at 추가
                ))
            
            # pbr_app_state.reviews_df 업데이트 로직
            if pbr_app_state.reviews_df is not None:
                mask = (pbr_app_state.reviews_df['user_id'] == user_id) & \
                    (pbr_app_state.reviews_df['content_id'] == content_id) & \
                    (pbr_app_state.reviews_df['type'] == content_type)
                if mask.any():
                    pbr_app_state.reviews_df.loc[mask, 'score'] = score
                    pbr_app_state.reviews_df.loc[mask, 'updated_at'] = pd.Timestamp.now()
                else:
                    new_row_df = pd.DataFrame([{
                        'user_id': user_id, 'content_id': content_id, 'type': content_type,
                        'score': score,
                        'review_content': None, # 기본값
                        'like_count': 0, 'report_count': 0, 'review_status': 'COMMON', # 기본값
                        'created_at': pd.Timestamp.now(), 'updated_at': pd.Timestamp.now()
                    }])
                    pbr_app_state.reviews_df = pd.concat([pbr_app_state.reviews_df, new_row_df], ignore_index=True)
            else:
                pbr_app_state.reviews_df = pd.DataFrame([{
                    'user_id': user_id, 'content_id': content_id, 'type': content_type,
                    'score': score,
                    'review_content': None,
                    'like_count': 0, 'report_count': 0, 'review_status': 'COMMON',
                    'created_at': pd.Timestamp.now(), 'updated_at': pd.Timestamp.now()
                }])
            logger.info(f"DB/State: 사용자 {user_id}의 평점({score}) 저장/업데이트 준비 완료.")
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="유효하지 않은 reaction_type입니다. '좋아요', '싫어요', '평점' 중 하나여야 합니다.")

        # 2. 페르소나 재계산 및 DB/pbr_app_state 업데이트 (핵심 변경 부분)
        # 피드백이 DB에 커밋된 후 페르소나 점수를 다시 계산하고 상태를 업데이트
        persona_generator.update_user_persona_scores(user_id, db)
        logger.info(f"사용자 {user_id}의 페르소나 점수 DB 및 메모리 상태 업데이트 요청 완료.")

        # 모든 DB 변경사항을 한 번에 커밋 (이것이 유일한 commit이어야 합니다!)
        db.commit()
        logger.info(f"DB: 사용자 {user_id} 피드백 및 페르소나 업데이트 관련 모든 변경사항 커밋 완료.")

    except Exception as e:
        db.rollback()
        logger.error(f"피드백 및 페르소나 업데이트 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"피드백 처리 중 오류 발생: {e}")

    # 3. 업데이트된 페르소나 정보를 기반으로 최종 추천 생성 및 반환
    unified_feedback = persona_generator.generate_unified_user_feedback(user_id, db)
    main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores = \
        persona_generator.get_hybrid_persona(user_id, unified_feedback, db)
    
    recommendations: List[RecommendedContent] = []

    # CF 추천 시도
    if pbr_app_state.user_item_matrix is not None and \
       pbr_app_state.user_item_matrix.shape[0] > 0 and \
       user_id in pbr_app_state.user_id_to_idx_map:
        recommendations = persona_generator.recommend_contents_cf(
            user_id, num_recommendations=RECOMMENDATION_COUNT, db=db, content_type_filter=content_type
            # main_persona_genres=main_persona_genres, sub_persona_genres=sub_persona_genres # 제거
        )
        recommendations.extend(recommendations)
        logger.info(f"CF 추천 {len(recommendations)}개 추가됨. 현재 총 추천 수: {len(recommendations)}")
    
    # CF 추천이 없거나 불가능한 경우 페르소나 기반 폴백
    if not recommendations and main_persona_id is not None:
        logger.info(f"CF 추천 실패/부족. 사용자 {user_id}의 페르소나 기반 추천을 시도합니다.")
        recommendations = persona_generator.get_persona_based_popular_fallback_recommendations( # _get_persona_based_popular_fallback_recommendations 함수명으로 통일
            user_id,
            main_persona_id,
            num_recommendations=RECOMMENDATION_COUNT,
            db=db,
            content_type_filter=content_type
            # main_persona_genres=main_persona_genres, sub_persona_genres=sub_persona_genres # 제거
        )
    
    # 모든 추천이 실패하거나 부족한 경우 일반 인기 추천
    if not recommendations:
        logger.info(f"페르소나 기반 추천 실패/부족. 사용자 {user_id}의 일반 인기 추천을 시도합니다.")
        recommendations = persona_generator._get_general_popular_recommendations(
            user_id, num_recommendations=RECOMMENDATION_COUNT, content_type_filter=content_type, db=db
            # main_persona_genres=main_persona_genres, sub_persona_genres=sub_persona_genres # 제거
        )

    logger.info(f"사용자 {user_id} 피드백 처리 완료. {len(recommendations)}개 추천.")
    return RecommendationResponse(
        message="피드백이 성공적으로 처리되었고 새로운 추천이 제공되었습니다.",
        recommendations=recommendations,
        main_persona=main_persona_name,
        sub_persona=sub_persona_name,
        all_personas_scores=all_personas_scores
    )


@persona_recommender_router.get("/users/{user_id}/recommendations", response_model=RecommendationResponse)
async def get_user_recommendations(
    user_id: int,
    content_type: Optional[str] = None,
    db: Session = Depends(get_db) # DB 세션을 의존성 주입으로 받습니다.
):
    """
    특정 사용자에게 추천 콘텐츠 목록을 제공합니다.
    """
    logger.info(f"사용자 {user_id}의 추천 요청 접수 (유형: {content_type}).")

    if pbr_app_state.contents_df.empty:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="추천 시스템 데이터가 아직 로드되지 않았거나 비어 있습니다.")

    try:
        # 사용자 페르소나 및 점수 업데이트 (필요시, 요청마다 업데이트)
        persona_generator.update_user_persona_scores(user_id, db)
        logger.info(f"사용자 {user_id}의 최신 페르소나 점수를 DB에 업데이트 시도.")
        db.commit() # 여기서도 commit 필요

        unified_feedback = persona_generator.generate_unified_user_feedback(user_id, db)
        main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores = \
            persona_generator.get_hybrid_persona(user_id, unified_feedback, db)

        recommendations: List[RecommendedContent] = []
        existing_content_ids = set() # 중복 추천 방지를 위한 셋

        # 1단계: CF 추천 시도
        if pbr_app_state.user_item_matrix is not None and \
           pbr_app_state.user_item_matrix.shape[0] > 0 and \
           user_id in pbr_app_state.user_id_to_idx_map:
            
            # CF 추천 요청 (최대 RECOMMENDATION_COUNT 개)
            cf_recs = persona_generator.recommend_contents_cf(
                user_id, num_recommendations=RECOMMENDATION_COUNT, db=db, content_type_filter=content_type # <--- db=db 추가
            )
            # 중복 제거 후 추가
            for rec in cf_recs:
                if rec.contentId not in existing_content_ids: # contentId로 통일
                    recommendations.append(rec)
                    existing_content_ids.add(rec.contentId) # contentId로 통일
                    if len(recommendations) >= RECOMMENDATION_COUNT: # 목표 개수에 도달하면 중단
                        break
            logger.info(f"CF 추천 시도 후 {len(cf_recs)}개 중 {len(recommendations)}개 추가됨. 현재 총 추천 수: {len(recommendations)}")
        
        # 2단계: CF 추천이 부족하면 페르소나 기반 폴백으로 채우기
        if len(recommendations) < RECOMMENDATION_COUNT and main_persona_id is not None:
            remaining_count = RECOMMENDATION_COUNT - len(recommendations)
            logger.info(f"CF 추천 부족 ({len(recommendations)}/{RECOMMENDATION_COUNT}). 페르소나 기반 추천으로 {remaining_count}개 채움 시도.")
            
            persona_recs = persona_generator.get_persona_based_popular_fallback_recommendations(
                user_id,
                main_persona_id,
                num_recommendations=remaining_count, # 남은 개수만큼 요청
                db=db, # <--- db=db 추가
                content_type_filter=content_type
            )
            # 중복 제거 후 추가
            added_from_persona = 0
            for rec in persona_recs:
                if rec.contentId not in existing_content_ids: # contentId로 통일
                    recommendations.append(rec)
                    existing_content_ids.add(rec.contentId) # contentId로 통일
                    added_from_persona += 1
                    if len(recommendations) >= RECOMMENDATION_COUNT: # 목표 개수에 도달하면 중단
                        break
            logger.info(f"페르소나 기반 추천 시도 후 {len(persona_recs)}개 중 {added_from_persona}개 추가됨. 현재 총 추천 수: {len(recommendations)}")


        # 3단계: 모든 추천이 부족하면 일반 인기 추천으로 채우기
        if len(recommendations) < RECOMMENDATION_COUNT:
            remaining_count = RECOMMENDATION_COUNT - len(recommendations)
            logger.info(f"페르소나 기반 추천 부족 ({len(recommendations)}/{RECOMMENDATION_COUNT}). 일반 인기 추천으로 {remaining_count}개 채움 시도.")
            
            # 일반 인기 추천을 remaining_count보다 더 많이 요청 (예: *3 정도로 넉넉하게)
            general_recs = persona_generator._get_general_popular_recommendations(
                user_id, num_recommendations=remaining_count * 3, content_type_filter=content_type, db=db
            )

            # 중복 제거 후 추가
            added_from_general = 0
            for rec in general_recs:
                if rec.contentId not in existing_content_ids: # contentId로 통일
                    recommendations.append(rec)
                    existing_content_ids.add(rec.contentId) # contentId로 통일
                    added_from_general += 1
                    if len(recommendations) >= RECOMMENDATION_COUNT: # 목표 개수에 도달하면 중단
                        break
            logger.info(f"일반 인기 추천 시도 후 {len(general_recs)}개 중 {added_from_general}개 추가됨. 현재 총 추천 수: {len(recommendations)}")

        # 최종적으로 정확히 RECOMMENDATION_COUNT 개만 반환하도록 자르기
        final_recommendations = recommendations[:RECOMMENDATION_COUNT]
        logger.info(f"사용자 {user_id} 추천 요청 처리 완료. 최종 {len(final_recommendations)}개 추천 반환.")

        return RecommendationResponse(
            message="추천 목록이 성공적으로 제공되었습니다.",
            recommendations=final_recommendations,
            main_persona=main_persona_name,
            sub_persona=sub_persona_name,
            all_personas_scores=all_personas_scores
        )

    except Exception as e:
        db.rollback() # 오류 발생 시 롤백
        logger.error(f"추천 생성 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"추천 생성 중 오류 발생: {e}")

@persona_recommender_router.get("/persona-counts", response_model=PersonaCountsResponse)
async def get_persona_counts(
    db: Session = Depends(get_db)
):
    """
    각 페르소나에 속하는 사용자 수를 반환합니다.
    """
    logger.info("페르소나별 사용자 수 계산 요청 접수.")

    if pbr_app_state.all_user_personas_df.empty:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="페르소나 데이터가 아직 로드되지 않았거나 비어 있습니다.")

    main_persona_ids_per_user = pbr_app_state.all_user_personas_df.loc[
        pbr_app_state.all_user_personas_df.groupby('user_id')['score'].idxmax()
    ]['persona_id']

    persona_counts = main_persona_ids_per_user.value_counts().to_dict()

    persona_name_counts: Dict[str, int] = {}
    if pbr_app_state.persona_id_to_name_map: # 매핑 정보가 있는지 확인
        for persona_id, count in persona_counts.items():
            persona_name = pbr_app_state.persona_id_to_name_map.get(persona_id, f"({persona_id})")
            persona_name_counts[persona_name] = int(count)
    else:
        logger.warning("pbr_app_state.persona_id_to_name_map이 초기화되지 않아 페르소나 ID를 이름으로 변환할 수 없습니다.")
        # 매핑이 없으면 ID를 그대로 사용하거나 오류 처리
        persona_name_counts = {f"ID_{pid}": int(count) for pid, count in persona_counts.items()}
    
    total_users_with_persona = len(main_persona_ids_per_user)

    logger.info("페르소나별 사용자 수 계산 완료.")
    return PersonaCountsResponse(
        message="페르소나별 사용자 수입니다.",
        persona_user_counts=persona_name_counts,
        total_users_with_persona=total_users_with_persona
    )