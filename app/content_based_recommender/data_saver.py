from sqlalchemy.orm import Session
from sqlalchemy import and_
from pydantic import BaseModel
from typing import List, Optional
import traceback
from datetime import date, timedelta
from models import ContentRecommendation, Content, PopularContent, ContentReaction
from content_based_recommender.schemas import ContentResponse, ContentRecommendationListResponse, RecommendationListResponse

def get_top_ranked_content(session: Session, batch_type: str = None) -> tuple[int, str] | None:

    today = date.today()
    yesterday = today - timedelta(days=1)

    for target_date in [today, yesterday]:
        query = session.query(PopularContent).filter(
            PopularContent.ranked_date == target_date,
            PopularContent.ranking == 1
        )
        if batch_type:
            query = query.filter(PopularContent.batch_type == batch_type.upper())

        top_content = query.first()

        if top_content:
            return top_content.content_id, top_content.content_type

    return None

# def get_existing_recommendations(db: Session, source_content_id: int, source_content_type: str, user_id: int) -> List[ContentResponse]:
#     """
#     DB에서 기존 추천 데이터를 조회
#     """
#     try:
#         # ranking 순서로 정렬하여 추천 데이터 조회
#         recommendations = db.query(ContentRecommendation).join(
#             Content,
#             and_(
#                 ContentRecommendation.recommended_content_id == Content.id,
#                 ContentRecommendation.recommended_content_type == Content.type
#             )
#         ).filter(
#             and_(
#                 ContentRecommendation.source_content_id == source_content_id,
#                 ContentRecommendation.source_content_type == source_content_type
#             )
#         ).order_by(ContentRecommendation.ranking).all()
        
#         if not recommendations:
#             return []
        
#         # 응답 형태로 변환
#         result = []
#         for rec in recommendations:
#             content_response = ContentResponse(
#                 content_id=rec.recommended_content.id,
#                 content_type=rec.recommended_content.type,
#                 title=rec.recommended_content.title,
#                 poster_path=rec.recommended_content.poster_path
#             )
#             result.append(content_response)
        
#         return result
#     except Exception as e:
#         print(f"DB 조회 중 오류 발생: {str(e)}")
#         traceback.print_exc()
#         return []

def get_existing_recommendations(
    db: Session,
    source_content_id: int,
    source_content_type: str,
    user_id: Optional[int]
) -> List[ContentResponse]:
    """
    DB에서 기존 추천 데이터를 조회하고, user_id가 주어진 경우 사용자의 반응(Like/Dislike)도 포함하여 반환.
    """
    try:
        # 추천된 콘텐츠 목록 조회
        recommendations = db.query(ContentRecommendation).join(
            Content,
            and_(
                ContentRecommendation.recommended_content_id == Content.id,
                ContentRecommendation.recommended_content_type == Content.type
            )
        ).filter(
            and_(
                ContentRecommendation.source_content_id == source_content_id,
                ContentRecommendation.source_content_type == source_content_type
            )
        ).order_by(ContentRecommendation.ranking).all()

        if not recommendations:
            return []

        result = []
        for rec in recommendations:
            recommended = rec.recommended_content

            user_reaction = None

            if user_id is not None:
                reaction_entry = db.query(ContentReaction).filter(
                    and_(
                        ContentReaction.user_id == user_id,
                        ContentReaction.content_id == recommended.id,
                        ContentReaction.type == recommended.type
                    )
                ).first()

                if reaction_entry:
                    user_reaction = reaction_entry.reaction.name  # Enum → 문자열 변환 (예: 'Like' 또는 'Dislike')

            result.append(ContentResponse(
                content_id=recommended.id,
                content_type=recommended.type,
                title=recommended.title,
                poster_path=recommended.poster_path,
                user_reaction=user_reaction
            ))

        return result

    except Exception as e:
        print(f"DB 조회 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return []


def save_recommendations_to_db(db: Session, source_content_id: int, source_content_type: str, 
                              recommendations: List[ContentRecommendationListResponse]) -> bool:
    try:
        # 기존 추천 데이터 삭제 (업데이트를 위해)
        db.query(ContentRecommendation).filter(
            and_(
                ContentRecommendation.source_content_id == source_content_id,
                ContentRecommendation.source_content_type == source_content_type
            )
        ).delete()
        
        # 새로운 추천 데이터 저장
        for idx, rec in enumerate(recommendations, 1):
            # Content 테이블에 추천된 콘텐츠가 있는지 확인하고 없으면 생성
            existing_content = db.query(Content).filter(
                and_(
                    Content.id == rec.content_id,
                    Content.type == rec.content_type
                )
            ).first()
            
            if not existing_content:
                # Content 데이터 생성 (추천 시스템에서 poster_path도 제공하므로 포함)
                new_content = Content(
                    id=rec.content_id,
                    type=rec.content_type,
                    title=rec.title,
                    overview=rec.overview
                    # poster_path는 추천 시스템 결과에 포함되어 있지만 RecommendationResponse에는 없음
                    # 필요하다면 별도로 추가 처리 필요
                )
                db.add(new_content)
            
            # ContentRecommendation 데이터 생성
            recommendation_record = ContentRecommendation(
                source_content_id=source_content_id,
                source_content_type=source_content_type,
                recommended_content_id=rec.content_id,
                recommended_content_type=rec.content_type,
                ranking=idx,
                score=rec.total_similarity
            )
            db.add(recommendation_record)
        
        db.commit()
        return True
    except Exception as e:
        db.rollback()
        print(f"DB 저장 중 오류 발생: {str(e)}")
        traceback.print_exc()
        return False

def build_recommendation_responses(
    db: Session,
    recommendations: List[ContentRecommendationListResponse],
    user_id: Optional[int] = None
) -> RecommendationListResponse:
    """
    추천 결과를 ContentResponse 형태로 변환. user_id가 있으면 반응 포함.
    """
    content_responses = []

    for rec in recommendations:
        user_reaction = None

        if user_id is not None:
            reaction_entry = db.query(ContentReaction).filter(
                and_(
                    ContentReaction.user_id == user_id,
                    ContentReaction.content_id == rec.content_id,
                    ContentReaction.type == rec.content_type
                )
            ).first()

            if reaction_entry:
                user_reaction = reaction_entry.reaction.name  # Like / Dislike

        content_response = ContentResponse(
            content_id=rec.content_id,
            content_type=rec.content_type,
            title=rec.title,
            poster_path=rec.poster_path,
            user_reaction=user_reaction
        )
        content_responses.append(content_response)

    return RecommendationListResponse(recommendations=content_responses)
