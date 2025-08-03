import logging
from fastapi import APIRouter, HTTPException, Depends, status
from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
from fastapi import Path

from database import get_db
from models import ContentReaction, Review, ReactionType, User

from . import persona_generator
from .state import pbr_app_state
from .config import (
    RECOMMENDATION_COUNT
)
from persona_based_recommender.persona_generator import (
    calculate_and_store_user_personas,
    update_user_persona_scores   
)
from .schemas import InitialFeedbackRequest, FeedbackRequest, RecommendationResponse, RecommendedContent, PersonaCountsResponse


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

persona_recommender_router = APIRouter(prefix="/recommends/personas")


@persona_recommender_router.post("/onboard", response_model=RecommendationResponse, status_code=status.HTTP_200_OK, summary="새로운 사용자 페르소나 부여")
async def onboard_user(
    request: InitialFeedbackRequest,
    db: Session = Depends(get_db)
):
    """
    신규 사용자 온보딩: 초기 피드백(선호 콘텐츠 좋아요)과 QA 답변을 기반으로 페르소나를 결정하고 추천을 제공합니다.
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
                email=f"user_{user_id}@example.com", 
                name=f"User {user_id}",              
                password="hashed_password",        
                is_active=True
            )
            db.add(new_user)
            db.flush()
            logger.info(f"DB: 사용자 {user_id}를 users 테이블에 새로 생성했습니다.")
        except Exception as e:
            db.rollback()
            logger.error(f"사용자 {user_id} 생성 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="사용자 생성 실패.")

    new_reactions = []
    for item in request.feedback_items:
        content_exists = pbr_app_state.contents_df[
            (pbr_app_state.contents_df['content_id'] == item.content_id) &
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

        try:
            db.commit()
            logger.info(f"DB: 사용자 {user_id}의 초기 콘텐츠 반응 {len(new_reactions)}개 DB에 저장 완료 (초기 커밋).")
        except Exception as e:
            db.rollback()
            logger.error(f"사용자 {user_id}의 초기 반응 저장 중 오류 발생: {e}", exc_info=True)
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="초기 피드백 저장 실패.")
        

        new_reactions_df_data = []
        for r in new_reactions:
            new_reactions_df_data.append({
                'user_id': r.user_id,
                'content_id': r.content_id,
                'type': r.type,
                'reaction': r.reaction.value
            })
        new_reactions_df = pd.DataFrame(new_reactions_df_data)

        if pbr_app_state.reactions_df is not None and not pbr_app_state.reactions_df.empty:

            existing_reactions_keys = pbr_app_state.reactions_df[
                (pbr_app_state.reactions_df['user_id'] == user_id) &
                (pbr_app_state.reactions_df['reaction'] == ReactionType.LIKE.value)
            ].set_index(['user_id', 'content_id', 'type']).index
            
            new_reactions_df_filtered = new_reactions_df[
                ~new_reactions_df.set_index(['user_id', 'content_id', 'type']).index.isin(
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
    
    unified_feedback = persona_generator.generate_unified_user_feedback(user_id, db)

    main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores_by_name = \
        persona_generator.get_hybrid_persona(user_id, unified_feedback, db)

    calculated_persona_scores_by_id = {}
    for name, score in all_personas_scores_by_name.items():
        normalized_name = name.strip()
        if normalized_name in pbr_app_state.persona_name_to_id_map:
            persona_id = pbr_app_state.persona_name_to_id_map[normalized_name]
            calculated_persona_scores_by_id[persona_id] = score
        else:
            logger.warning(
                f"경고: 온보딩 중 페르소나 이름 '{name}' (정규화: '{normalized_name}')이(가) "
                f"pbr_app_state.persona_name_to_id_map에서 찾아지지 않아 해당 점수는 저장되지 않습니다. "
                f"현재 매핑된 키: {list(pbr_app_state.persona_name_to_id_map.keys())}"
            )

    calculate_and_store_user_personas(db, user_id=user_id, calculated_persona_scores=calculated_persona_scores_by_id)
    
    try:
        db.commit()
        logger.info(f"DB: 사용자 {user_id} 온보딩 관련 모든 변경사항 커밋 완료.")

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

    except Exception as e:
        db.rollback()
        logger.error(f"사용자 {user_id} 온보딩 중 DB 커밋 또는 메모리 갱신 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="DB 저장 중 오류 발생.")

    recommendations: List[RecommendedContent] = []

    if main_persona_id is not None and main_persona_name != "Undetermined":
        logger.info(f"사용자 {user_id}의 메인 페르소나({main_persona_name})를 기반으로 추천을 생성합니다.")
        recommendations = persona_generator.update_and_get_recommendations( 
            user_id=user_id,
            db=db,
            num_recommendations=RECOMMENDATION_COUNT,
            content_type_filter=None
        )
    else:
        logger.info(f"사용자 {user_id}의 페르소나 기반 추천 생성 불가 (페르소나 미결정), 일반 인기 추천으로 대체.")
        recommendations = persona_generator._get_general_popular_recommendations(user_id, num_recommendations=RECOMMENDATION_COUNT, db=db)

    logger.info(f"사용자 {user_id} 온보딩 완료. {len(recommendations)}개 추천.")

    normalized_and_named_scores = {}
    if all_personas_scores_by_name:
        total_score = sum(all_personas_scores_by_name.values())
        
        persona_id_to_name = pbr_app_state.persona_df.set_index('persona_id')['name'].to_dict()

        for persona_id, score in all_personas_scores_by_name.items():
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


@persona_recommender_router.post("/feedback", response_model=RecommendationResponse, status_code=status.HTTP_200_OK, summary="기존 사용자 좋아요, 싫어요, 별점 평가 & 페르소나 점수 업데이트")
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
    reaction_type_str = request.reaction_type 
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
        if reaction_type_str in ['좋아요', '싫어요']:
            db_reaction_enum = None
            if reaction_type_str == '좋아요':
                db_reaction_enum = ReactionType.LIKE
            elif reaction_type_str == '싫어요':
                db_reaction_enum = ReactionType.DISLIKE
            
            if db_reaction_enum is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="유효하지 않은 반응 유형입니다.")

            existing_reaction = db.query(ContentReaction).filter_by(
                user_id=user_id, content_id=content_id, type=content_type
            ).first()
            if existing_reaction:
                existing_reaction.reaction = db_reaction_enum
                db.add(existing_reaction)
            else:
                db.add(ContentReaction(user_id=user_id, content_id=content_id, type=content_type, reaction=db_reaction_enum))
            
            if pbr_app_state.reactions_df is not None:
                mask = (pbr_app_state.reactions_df['user_id'] == user_id) & \
                       (pbr_app_state.reactions_df['content_id'] == content_id) & \
                       (pbr_app_state.reactions_df['type'] == content_type)
                if mask.any():
                    pbr_app_state.reactions_df.loc[mask, 'reaction'] = db_reaction_enum.value
                else:
                    new_row_df = pd.DataFrame([{
                        'user_id': user_id, 'content_id': content_id, 'type': content_type,
                        'reaction': db_reaction_enum.value
                    }])
                    pbr_app_state.reactions_df = pd.concat([pbr_app_state.reactions_df, new_row_df], ignore_index=True)
            else:
                pbr_app_state.reactions_df = pd.DataFrame([{
                    'user_id': user_id, 'content_id': content_id, 'type': content_type,
                    'reaction': db_reaction_enum.value,
                    'created_at': pd.Timestamp.now(), 'updated_at': pd.Timestamp.now()
                }])
            logger.info(f"DB/State: 사용자 {user_id}의 콘텐츠 반응({reaction_type_str}) 저장/업데이트 준비 완료.")
            db.flush()

        elif reaction_type_str == '평점':
            if score is None:
                raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="평점 반응에는 'score'가 필수입니다.")
            
            existing_review = db.query(Review).filter_by(
                user_id=user_id, content_id=content_id, type=content_type
            ).first()
            if existing_review:
                existing_review.score = score
                db.add(existing_review)
            else:
                db.add(Review(
                    user_id=user_id,
                    content_id=content_id,
                    type=content_type,
                    score=score
                ))
            
            if pbr_app_state.reviews_df is not None:
                mask = (pbr_app_state.reviews_df['user_id'] == user_id) & \
                    (pbr_app_state.reviews_df['content_id'] == content_id) & \
                    (pbr_app_state.reviews_df['type'] == content_type)
                if mask.any():
                    pbr_app_state.reviews_df.loc[mask, 'score'] = score
                else:
                    new_row_df = pd.DataFrame([{
                        'user_id': user_id, 'content_id': content_id, 'type': content_type,
                        'score': score,
                        'review_content': None,
                        'like_count': 0, 'report_count': 0, 'review_status': 'COMMON'
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
            db.flush()
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="유효하지 않은 reaction_type입니다. '좋아요', '싫어요', '평점' 중 하나여야 합니다.")

        persona_generator.update_user_persona_scores(user_id, db)
        logger.info(f"사용자 {user_id}의 페르소나 점수 DB 및 메모리 상태 업데이트 요청 완료.")
        
    except Exception as e:
        db.rollback()
        logger.error(f"피드백 및 페르소나 업데이트 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"피드백 처리 중 오류 발생: {e}")

    unified_feedback = persona_generator.generate_unified_user_feedback(user_id, db)
    main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores = \
        persona_generator.get_hybrid_persona(user_id, unified_feedback, db)
    
    recommendations: List[RecommendedContent] = []

    if pbr_app_state.user_item_matrix is not None and \
       pbr_app_state.user_item_matrix.shape[0] > 0 and \
       user_id in pbr_app_state.user_id_to_idx_map:
        recommendations = persona_generator.recommend_contents_cf(
            user_id, num_recommendations=RECOMMENDATION_COUNT, db=db, content_type_filter=content_type
        )
        logger.info(f"CF 추천 {len(recommendations)}개 추가됨. 현재 총 추천 수: {len(recommendations)}")
    
    if not recommendations and main_persona_id is not None:
        logger.info(f"CF 추천 실패/부족. 사용자 {user_id}의 페르소나 기반 추천을 시도합니다.")
        recommendations = persona_generator.get_persona_based_popular_fallback_recommendations( 
            user_id,
            main_persona_id,
            num_recommendations=RECOMMENDATION_COUNT,
            db=db,
            content_type_filter=content_type
        )
    
    if not recommendations:
        logger.info(f"페르소나 기반 추천 실패/부족. 사용자 {user_id}의 일반 인기 추천을 시도합니다.")
        recommendations = persona_generator._get_general_popular_recommendations(
            user_id, num_recommendations=RECOMMENDATION_COUNT, content_type_filter=content_type, db=db
        )

    logger.info(f"사용자 {user_id} 피드백 처리 완료. {len(recommendations)}개 추천.")

    db.commit()
    logger.info(f"DB: 사용자 {user_id} 피드백 및 페르소나 업데이트 관련 모든 변경사항 최종 커밋 완료.")

    return RecommendationResponse(
        message="피드백이 성공적으로 처리되었고 새로운 추천이 제공되었습니다.",
        recommendations=recommendations,
        main_persona=main_persona_name,
        sub_persona=sub_persona_name,
        all_personas_scores=all_personas_scores
    )

@persona_recommender_router.delete("/users/{user_id}/contents/{content_id}/reaction/{content_type}", status_code=status.HTTP_204_NO_CONTENT, summary="좋아요, 싫어요 취소")
async def delete_content_reaction(
    user_id: int = Path(..., description="사용자 ID"), 
    content_id: int = Path(..., description="콘텐츠 ID"),
    content_type: str = Path(..., description="콘텐츠 타입 (예: movie, tv)"), 
    db: Session = Depends(get_db)
):
    """
    사용자의 특정 콘텐츠 반응(좋아요/싫어요)을 삭제합니다. 사용자 페르소나 점수도 업데이트됩니다.
    """
    existing_reaction = db.query(ContentReaction).filter(
        ContentReaction.user_id == user_id,
        ContentReaction.content_id == content_id,
        ContentReaction.type == content_type 
    ).first()

    if not existing_reaction:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="해당 반응을 찾을 수 없습니다.")

    try:
        db.delete(existing_reaction)
        db.commit()
        logger.info(f"사용자 {user_id}의 콘텐츠 {content_id} ({content_type}) 반응 삭제 완료.")
        
        update_user_persona_scores(user_id, db)

    except Exception as e:
        db.rollback()
        logger.error(f"반응 삭제 중 오류 발생: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="반응 삭제 중 오류가 발생했습니다.")


@persona_recommender_router.delete("/users/{user_id}/reviews/{review_id}", status_code=status.HTTP_204_NO_CONTENT, summary="리뷰 삭제")
async def delete_review(user_id: int, review_id: int, db: Session = Depends(get_db)):
    """
    사용자의 특정 리뷰를 삭제합니다. 사용자 페르소나 점수도 업데이트됩니다.
    """
    existing_review = db.query(Review).filter(
        Review.review_id == review_id,
        Review.user_id == user_id 
    ).first()

    if not existing_review:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="리뷰를 찾을 수 없습니다.")

    try:
        db.delete(existing_review)
        db.commit()
        logger.info(f"사용자 {user_id}의 리뷰 {review_id} 삭제 완료.")

        update_user_persona_scores(user_id, db)

    except Exception as e:
        db.rollback()
        logger.error(f"리뷰 삭제 중 오류 발생: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="리뷰 삭제 중 오류가 발생했습니다.")


@persona_recommender_router.get("/users/{user_id}/recommendations", response_model=RecommendationResponse, summary="사용자 최신 추천 리스트 및 페르소나 점수 확인")
async def get_user_recommendations(
    user_id: int,
    content_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    특정 사용자에게 추천 콘텐츠 목록을 제공합니다.
    """
    logger.info(f"사용자 {user_id}의 추천 요청 접수 (유형: {content_type}).")

    if pbr_app_state.contents_df.empty:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="추천 시스템 데이터가 아직 로드되지 않았거나 비어 있습니다.")

    try:
        unified_feedback = persona_generator.generate_unified_user_feedback(user_id, db)
        main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores = \
            persona_generator.get_hybrid_persona(user_id, unified_feedback, db)

        recommendations: List[RecommendedContent] = []
        existing_content_ids = set() 

        if pbr_app_state.user_item_matrix is not None and \
           pbr_app_state.user_item_matrix.shape[0] > 0 and \
           user_id in pbr_app_state.user_id_to_idx_map:
            
            cf_recs = persona_generator.recommend_contents_cf(
                user_id, num_recommendations=RECOMMENDATION_COUNT, db=db, content_type_filter=content_type
            )
            for rec in cf_recs:
                if rec.contentId not in existing_content_ids: 
                    recommendations.append(rec)
                    existing_content_ids.add(rec.contentId) 
                    if len(recommendations) >= RECOMMENDATION_COUNT: 
                        break
            logger.info(f"CF 추천 시도 후 {len(cf_recs)}개 중 {len(recommendations)}개 추가됨. 현재 총 추천 수: {len(recommendations)}")
        
        if len(recommendations) < RECOMMENDATION_COUNT and main_persona_id is not None:
            remaining_count = RECOMMENDATION_COUNT - len(recommendations)
            logger.info(f"CF 추천 부족 ({len(recommendations)}/{RECOMMENDATION_COUNT}). 페르소나 기반 추천으로 {remaining_count}개 채움 시도.")
            
            persona_recs = persona_generator.get_persona_based_popular_fallback_recommendations(
                user_id,
                main_persona_id,
                num_recommendations=remaining_count,
                db=db,
                content_type_filter=content_type
            )
            added_from_persona = 0
            for rec in persona_recs:
                if rec.contentId not in existing_content_ids:
                    recommendations.append(rec)
                    existing_content_ids.add(rec.contentId) 
                    added_from_persona += 1
                    if len(recommendations) >= RECOMMENDATION_COUNT: 
                        break
            logger.info(f"페르소나 기반 추천 시도 후 {len(persona_recs)}개 중 {added_from_persona}개 추가됨. 현재 총 추천 수: {len(recommendations)}")

        if len(recommendations) < RECOMMENDATION_COUNT:
            remaining_count = RECOMMENDATION_COUNT - len(recommendations)
            logger.info(f"페르소나 기반 추천 부족 ({len(recommendations)}/{RECOMMENDATION_COUNT}). 일반 인기 추천으로 {remaining_count}개 채움 시도.")
            
            general_recs = persona_generator._get_general_popular_recommendations(
                user_id, num_recommendations=remaining_count * 3, content_type_filter=content_type, db=db
            )

            added_from_general = 0
            for rec in general_recs:
                if rec.contentId not in existing_content_ids: 
                    recommendations.append(rec)
                    existing_content_ids.add(rec.contentId)
                    added_from_general += 1
                    if len(recommendations) >= RECOMMENDATION_COUNT:
                        break
            logger.info(f"일반 인기 추천 시도 후 {len(general_recs)}개 중 {added_from_general}개 추가됨. 현재 총 추천 수: {len(recommendations)}")

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
        db.rollback() 
        logger.error(f"추천 생성 중 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"추천 생성 중 오류 발생: {e}")

@persona_recommender_router.get("/persona-counts", response_model=PersonaCountsResponse, summary="각 페르소나에 속하는 사용자 수 반환")
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
    if pbr_app_state.persona_id_to_name_map: 
        for persona_id, count in persona_counts.items():
            persona_name = pbr_app_state.persona_id_to_name_map.get(persona_id, f"({persona_id})")
            persona_name_counts[persona_name] = int(count)
    else:
        logger.warning("pbr_app_state.persona_id_to_name_map이 초기화되지 않아 페르소나 ID를 이름으로 변환할 수 없습니다.")
        persona_name_counts = {f"ID_{pid}": int(count) for pid, count in persona_counts.items()}
    
    total_users_with_persona = len(main_persona_ids_per_user)

    logger.info("페르소나별 사용자 수 계산 완료.")
    return PersonaCountsResponse(
        message="페르소나별 사용자 수입니다.",
        persona_user_counts=persona_name_counts,
        total_users_with_persona=total_users_with_persona
    )