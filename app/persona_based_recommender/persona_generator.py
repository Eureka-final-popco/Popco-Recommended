import logging
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
from collections import defaultdict
from decimal import Decimal, getcontext, ROUND_DOWN

from sqlalchemy.orm import Session
from sqlalchemy import func, text

from .state import pbr_app_state
from .config import (
    LIKE_WEIGHT, DISLIKE_WEIGHT, STAR_RATING_WEIGHTS,
    CF_SIMILARITY_THRESHOLD,CONTENT_FEEDBACK_WEIGHT,
    MAX_EXPECTED_PERSONA_SCORE, MIN_PERSONA_SCORE, RECOMMENDATION_COUNT
)
from .schemas import RecommendedContent
from models import UserPersona, ContentReaction, Review, Persona, ReactionType 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

getcontext().prec = 20

def _create_recommended_content(
    content_data: Dict,
    predicted_rating: Optional[float] = None,
    persona_genre_match: Optional[float] = None
) -> RecommendedContent:
    """
    주어진 콘텐츠 데이터 딕셔너리를 기반으로 RecommendedContent 객체를 생성합니다.
    """
    genres_data = content_data.get('genres')
    genres_list = []
    if isinstance(genres_data, list):
        genres_list = [g.strip() for g in genres_data if isinstance(g, str) and g.strip()]
    elif isinstance(genres_data, str) and genres_data.strip():
        if ',' in genres_data:
            genres_list = [genre.strip() for genre in genres_data.split(',') if genre.strip()]
        elif '|' in genres_data:
            genres_list = [g.strip() for g in genres_data.split('|') if g.strip()]
        else:
            genres_list = [genres_data.strip()]


    return RecommendedContent(
        contentId=int(content_data.get('content_id')), 
        title=content_data.get('title', ''),
        genres=genres_list, 
        type=content_data.get('type', ''), 
        poster_path=content_data.get('poster_path', ''),
        predicted_rating=predicted_rating,
        persona_genre_match=persona_genre_match
    )


def calculate_user_content_matrix_sparse(
    reactions_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    user_id_to_idx_map: Dict[int, int],
    content_id_to_idx_map: Dict[Tuple[int, str], int]
) -> csr_matrix:
    logger.info("사용자-콘텐츠 희소 행렬 계산 시작...")

    num_users = len(user_id_to_idx_map)
    num_contents = len(content_id_to_idx_map)

    user_item_matrix = lil_matrix((num_users, num_contents))

    for _, row in reactions_df.iterrows():
        user_id = row['user_id']
        content_id_orig = row['content_id']
        content_type = row['type']
        reaction_type = row['reaction_type']

        user_idx = user_id_to_idx_map.get(user_id)
        content_tuple = (content_id_orig, content_type)
        content_idx = content_id_to_idx_map.get(content_tuple)

        if user_idx is not None and content_idx is not None:
            score = 0
            if reaction_type == 'liked':
                score = 1
            elif reaction_type == 'disliked':
                score = -1
            
            user_item_matrix[user_idx, content_idx] += score

    for _, row in reviews_df.iterrows():
        user_id = row['user_id']
        content_id_orig = row['content_id'] 
        content_type = row['content_type']
        rating = row['rating']

        user_idx = user_id_to_idx_map.get(user_id)
        content_tuple = (content_id_orig, content_type)
        content_idx = content_id_to_idx_map.get(content_tuple)

        if user_idx is not None and content_idx is not None:
            user_item_matrix[user_idx, content_idx] += rating 

    user_item_matrix_csr = user_item_matrix.tocsr()
    logger.info(f"사용자-콘텐츠 희소 행렬 계산 완료. Shape: {user_item_matrix_csr.shape}")
    return user_item_matrix_csr

def calculate_and_store_user_personas(
    db: Session,
    user_id: int,
    calculated_persona_scores: Dict[int, float]
):
    """
    계산된 사용자 페르소나 점수를 user_personas 테이블에 저장하거나 업데이트합니다.
    """
    logger.info(f"사용자 [{user_id}]의 페르소나 점수 DB 및 상태 저장 시작 (Upsert 방식).")

    if not calculated_persona_scores:
        logger.warning(f"사용자 {user_id}에 대해 저장할 페르소나 점수가 없습니다. 기존 페르소나도 삭제하지 않습니다.")
        return

    current_utc_time = datetime.utcnow()

    existing_user_personas = db.query(UserPersona).filter(UserPersona.user_id == user_id).all()
    existing_personas_map = {up.persona_id: up for up in existing_user_personas}

    persona_ids_to_keep = set()

    for persona_id, new_score in calculated_persona_scores.items():
        persona_name = pbr_app_state.persona_id_to_name_map.get(persona_id) 
        
        persona = db.query(Persona).filter(Persona.persona_id == persona_id).first()
        if not persona:
            persona_name_for_log = pbr_app_state.persona_id_to_name_map.get(persona_id, f"Unknown ID {persona_id}")
            logger.warning(f"DB Persona 테이블에서 페르소나 ID {persona_id} (이름: '{persona_name_for_log}')를 찾을 수 없습니다. 해당 페르소나 점수는 저장되지 않습니다.")
            continue

        persona_ids_to_keep.add(persona_id)
        if persona_id in existing_personas_map:
            user_persona = existing_personas_map[persona_id]
            score_changed = (user_persona.score != new_score)

            if score_changed:
                user_persona.score = new_score
                logger.debug(f"사용자 {user_id}, 페르소나 '{persona_name}' (ID: {persona_id}): 점수 업데이트됨 ({user_persona.score} -> {new_score}).")
            else:
                logger.debug(f"사용자 {user_id}, 페르소나 '{persona_name}' (ID: {persona_id}): 점수 변화 없음 ({new_score}).")

            user_persona.updated_at = current_utc_time

        else:
            new_user_persona = UserPersona(
                user_id=user_id,
                persona_id=persona_id,
                score=new_score,
                created_at=current_utc_time,
                updated_at=current_utc_time
            )
            db.add(new_user_persona)
            logger.debug(f"사용자 {user_id}, 페르소나 '{persona_name}' (ID: {persona_id}): 새로운 페르소나 점수 삽입됨 ({new_score}).")

    personas_to_delete = [
        up for up in existing_user_personas
        if up.persona_id not in persona_ids_to_keep
    ]

    if personas_to_delete:
        for ptd in personas_to_delete:
            db.delete(ptd)
        logger.info(f"사용자 {user_id}의 더 이상 유효하지 않은 페르소나 점수 {len(personas_to_delete)}개 삭제됨.")

    try:
        db.flush()
        logger.info(f"DB: 사용자 [{user_id}]의 페르소나 점수 변경사항 플러시 완료.")

        db.commit()
        logger.info(f"DB: 사용자 [{user_id}]의 페르소나 점수 변경사항 커밋 완료.")
    except Exception as e:
        db.rollback()
        logger.error(f"DB 변경사항 플러시 중 오류 발생: {e}", exc_info=True)
        raise

    logger.info(f"사용자 {user_id}의 페르소나 점수 DB 저장 완료 (Upsert 방식).")


def calculate_user_similarity(user_item_matrix: csr_matrix, user_idx_to_id_map: Dict[int, int]) -> Tuple[pd.DataFrame, csr_matrix]:
    logger.info("사용자 유사도 계산 시작...")
    if user_item_matrix.shape[0] == 0:
        logger.warning("user_item_matrix가 비어 있어 사용자 유사도를 계산할 수 없습니다.")
        return pd.DataFrame(columns=['user1_idx', 'user2_idx', 'similarity_score']), csr_matrix((0,0))

    user_similarity_matrix = cosine_similarity(user_item_matrix)
    logger.info(f"user_similarity_matrix shape: {user_similarity_matrix.shape}")

    similarity_data = []
    num_users = user_item_matrix.shape[0]
    for i in range(num_users):
        for j in range(i + 1, num_users): 
            similarity_score = user_similarity_matrix[i, j]
            if similarity_score > 0:
                similarity_data.append({
                    'user1_idx': i,
                    'user2_idx': j,
                    'similarity_score': similarity_score
                })

    final_similarity_df = pd.DataFrame(similarity_data)
    logger.info(f"사용자 유사도 계산 완료. 유사한 사용자 쌍: {len(final_similarity_df)}")
    
    user_similarity_matrix_sparse = csr_matrix(user_similarity_matrix)
    
    return final_similarity_df, user_similarity_matrix_sparse 


def calculate_content_similarity_sparse(user_item_matrix: csr_matrix) -> csr_matrix:
    logger.info("콘텐츠 유사도 계산 시작...")
    if user_item_matrix.shape[1] == 0: 
        logger.warning("user_item_matrix에 콘텐츠가 없어 콘텐츠 유사도를 계산할 수 없습니다.")
        return csr_matrix((0,0)) 

    content_similarity_matrix = cosine_similarity(user_item_matrix.T)
    logger.info(f"콘텐츠 유사도 계산 완료. Shape: {content_similarity_matrix.shape}")
    return csr_matrix(content_similarity_matrix)


def calculate_persona_similarity() -> pd.DataFrame:
    """
    페르소나 간의 유사도를 계산합니다.
    페르소나-장르 매핑을 기반으로 코사인 유사도를 사용합니다.
    """
    logger.info("페르소나 유사도 계산 중...")

    persona_df = pbr_app_state.persona_df
    persona_genres_df = pbr_app_state.persona_genres_df
    genres_df = pbr_app_state.genres_df

    if persona_df is None or persona_df.empty or \
       persona_genres_df is None or persona_genres_df.empty or \
       genres_df is None or genres_df.empty:
        logger.warning("페르소나 관련 데이터가 없어 페르소나 유사도를 계산할 수 없습니다. 빈 DataFrame을 반환합니다.")
        return pd.DataFrame(columns=['persona_id_1', 'persona_id_2', 'similarity_score'])

    persona_genre_merged = persona_genres_df.merge(
        genres_df, left_on='genre_id', right_on='id', suffixes=('_persona', '_genre')
    )

    persona_genre_matrix = persona_genre_merged.pivot_table(
        index='persona_id', columns='name', aggfunc='size', fill_value=0
    ).fillna(0)

    if persona_genre_matrix.empty:
        logger.warning("페르소나-장르 매트릭스가 비어 있습니다. 페르소나 유사도를 계산할 수 없습니다.")
        return pd.DataFrame(columns=['persona_id_1', 'persona_id_2', 'similarity_score'])

    persona_similarity = cosine_similarity(persona_genre_matrix)
    persona_similarity_df = pd.DataFrame(persona_similarity,
                                         index=persona_genre_matrix.index,
                                         columns=persona_genre_matrix.index)

    similarities = []
    for i in range(persona_similarity_df.shape[0]):
        for j in range(i + 1, persona_similarity_df.shape[1]): 
            persona_id_1 = persona_similarity_df.index[i]
            persona_id_2 = persona_similarity_df.columns[j]
            score = persona_similarity_df.iloc[i, j]
            similarities.append({
                'persona_id_1': int(persona_id_1),
                'persona_id_2': int(persona_id_2),
                'similarity_score': score
            })
    final_persona_similarity_df = pd.DataFrame(similarities)

    logger.info(f"페르소나 유사도 계산 완료. 결과 shape: {final_persona_similarity_df.shape}")
    logger.info(f"페르소나 유사도 통계:\n{final_persona_similarity_df['similarity_score'].describe()}")

    return final_persona_similarity_df


def generate_persona_keywords(persona_df: pd.DataFrame) -> Dict[int, List[str]]:
    """
    주어진 페르소나 데이터프레임에서 각 페르소나의 키워드를 생성하고 맵으로 반환합니다.
    """
    logger.info("페르소나 키워드 생성 시작...")
    
    persona_keyword_map: Dict[int, List[str]] = {}
    
    if pbr_app_state.persona_details_map:
        for persona_id, details in pbr_app_state.persona_details_map.items():
            if 'keywords' in details:
                persona_keyword_map[persona_id] = details['keywords']
    else:
        logger.warning("pbr_app_state.persona_details_map이 비어 있습니다. 키워드를 생성할 수 없습니다.")
        
    logger.info("페르소나 키워드 생성 완료.")
    return persona_keyword_map


def update_user_persona_scores(
    user_id: int,
    db: Session
):
    logger.info(f"사용자 {user_id}의 페르소나 점수 업데이트 시작.")

    unified_feedback = generate_unified_user_feedback(user_id, db)

    calculated_persona_scores_by_name = calculate_persona_scores_from_feedback(user_id, unified_feedback)

    logger.info(f"DEBUG: calculate_persona_scores_by_name 결과: {calculated_persona_scores_by_name}")
    
    calculated_persona_scores_by_id = {}
    for name, score in calculated_persona_scores_by_name.items():
        original_name = name 
        normalized_name = name.strip() 

        if normalized_name in pbr_app_state.persona_name_to_id_map:
            persona_id = pbr_app_state.persona_name_to_id_map[normalized_name]
            calculated_persona_scores_by_id[persona_id] = score
            logger.info(f"DEBUG: 페르소나 '{original_name}' (정규화: '{normalized_name}') -> ID {persona_id}로 성공적으로 매핑되었습니다.")
        else:
            logger.warning(f"경고: 페르소나 이름 '{original_name}' (정규화: '{normalized_name}')이(가) pbr_app_state.persona_name_to_id_map에서 찾아지지 않았습니다.")
            logger.warning(f"DEBUG: pbr_app_state.persona_name_to_id_map의 현재 키: {list(pbr_app_state.persona_name_to_id_map.keys())}")

    try:
        calculate_and_store_user_personas(db, user_id, calculated_persona_scores_by_id)
        logger.info(f"사용자 {user_id}의 페르소나 점수 DB 업데이트 완료.")
    except Exception as e:
        logger.error(f"사용자 {user_id}의 페르소나 점수 DB 저장 중 오류 발생: {e}", exc_info=True)

    logger.info(f"메모리 내 pbr_app_state.all_user_personas_df 및 user_persona_scores_map 갱신 시도.")
    try:
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
        
        logger.info(f"pbr_app_state.all_user_personas_df 갱신 완료. {len(pbr_app_state.all_user_personas_df)}개 레코드.")
        logger.info(f"pbr_app_state.user_persona_scores_map 갱신 완료. {len(pbr_app_state.user_persona_scores_map)}개 사용자.")

    except Exception as e:
        logger.error(f"메모리 내 pbr_app_state 데이터 갱신 중 오류 발생: {e}", exc_info=True)

    logger.info(f"사용자 {user_id}의 페르소나 점수 업데이트 처리 완료.")


def generate_unified_user_feedback(user_id: int, db: Session) -> Dict[Tuple[int, str], float]:
    """
    사용자의 좋아요/싫어요 및 리뷰 데이터를 통합하여 각 콘텐츠에 대한 피드백 점수를 생성합니다.
    이 함수는 User QA Answers (pbr_app_state.user_qa_answers_df)에 대한 가중치도 고려할 수 있습니다.
    ContentReaction 및 Review 객체에는 'content_id' 필드가 있다고 가정합니다.
    """
    logger.info(f"사용자 {user_id}의 통합 피드백 생성 중...")
    
    unified_feedback: Dict[Tuple[int, str], float] = defaultdict(float) 

    reactions = db.query(ContentReaction).filter(ContentReaction.user_id == user_id).all()
    for reaction in reactions:
        key = (reaction.content_id, reaction.type)
        if reaction.reaction == ReactionType.LIKE:
            unified_feedback[key] += LIKE_WEIGHT
        elif reaction.reaction == ReactionType.DISLIKE:
            unified_feedback[key] += DISLIKE_WEIGHT
    logger.info(f"ContentReaction {len(reactions)}개 처리 완료.")

    reviews = db.query(Review).filter(Review.user_id == user_id).all()
    for review in reviews:
        key = (review.content_id, review.type)
        score = STAR_RATING_WEIGHTS.get(review.score, 0.0)
        unified_feedback[key] += score
    logger.info(f"Review {len(reviews)}개 처리 완료.")

    logger.info(f"사용자 {user_id}의 통합 피드백 생성 완료. {len(unified_feedback)}개의 콘텐츠에 대한 피드백.")
    return unified_feedback


def _get_persona_genre_mapping(db: Session) -> Dict[int, List[str]]:
    """
    DB에서 페르소나-장르 매핑을 로드합니다.
    pbr_app_state.persona_genres_df와 pbr_app_state.genres_df를 사용하여 매핑을 생성합니다.
    결과: {persona_id: [genre_name1, genre_name2, ...]}
    """
    if pbr_app_state.persona_genres_df is None or pbr_app_state.genres_df is None:
        logger.warning("페르소나-장르 매핑을 위한 데이터프레임이 로드되지 않았습니다.")
        return {}

    persona_genre_map = defaultdict(list)
    
    genre_id_to_name = pbr_app_state.genres_df.set_index('id')['name'].to_dict()

    for _, row in pbr_app_state.persona_genres_df.iterrows():
        persona_id = row['persona_id']
        genre_id = row['genre_id']
        genre_name = genre_id_to_name.get(genre_id)
        if genre_name:
            persona_genre_map[persona_id].append(genre_name)
    
    return dict(persona_genre_map)


def calculate_persona_scores_from_feedback(
    user_id: int,
    unified_feedback: Dict[Tuple[int, str], float]
) -> Dict[str, float]:
    """
    사용자의 통합 피드백을 기반으로 각 페르소나에 대한 점수를 계산합니다.
    Args:
        user_id: 사용자 ID.
        unified_feedback: 사용자로부터 통합된 콘텐츠 피드백 (content_id -> score).
                          reaction과 review 점수가 통합된 딕셔너리입니다.
    Returns:
        각 페르소나의 이름에 대한 점수를 담은 딕셔너리 (persona_name -> score).
    """
    logger.info(f"사용자 {user_id}의 피드백 기반 페르소나 점수 계산 시작.")

    if pbr_app_state.persona_details_map is None or not pbr_app_state.persona_details_map:
        logger.error("페르소나 상세 정보(persona_details_map)가 로드되지 않았습니다.")
        return {}

    persona_scores: Dict[str, float] = defaultdict(float) 

    user_positive_genres: Set[str] = set()
    if unified_feedback: 
        logger.info(f"사용자 {user_id}의 콘텐츠 피드백 기반 페르소나 점수 계산 시작.")
        
        if 'content_id' not in pbr_app_state.contents_df.columns or \
           not pd.api.types.is_numeric_dtype(pbr_app_state.contents_df['content_id']):
            pbr_app_state.contents_df['content_id'] = pd.to_numeric(
                pbr_app_state.contents_df['content_id'], errors='coerce'
            )
            logger.warning("pbr_app_state.contents_df['content_id']를 'content_id'로 변환했습니다.")

        for (content_id, content_type), feedback_score in unified_feedback.items():
            if feedback_score > 0: 
                content_row = pbr_app_state.contents_df[
                    (pbr_app_state.contents_df['content_id'] == content_id) & 
                    (pbr_app_state.contents_df['type'] == content_type)
                ]
                
                if not content_row.empty: 
                    genres_data = content_row['genres'].iloc[0]
                    
                    if isinstance(genres_data, str):
                        genres_list = [g.strip() for g in genres_data.split('|') if g.strip()]
                    elif isinstance(genres_data, list):
                        genres_list = [g.strip() for g in genres_data if isinstance(g, str) and g.strip()]
                    else:
                        genres_list = []
                        logger.warning(f"콘텐츠 {content_id}의 장르 데이터 형식이 예상과 다릅니다: {type(genres_data)}. 스킵합니다.")

                    if genres_list:
                        user_positive_genres.update(genres_list)

                    for genre_name in genres_list:
                        genre_id = pbr_app_state.genre_name_to_id_map.get(genre_name)
                        if genre_id is not None:
                            relevant_personas_from_genre = pbr_app_state.persona_genres_df[
                                pbr_app_state.persona_genres_df['genre_id'] == genre_id
                            ]['persona_id'].tolist()

                            for p_id in relevant_personas_from_genre:
                                persona_name = pbr_app_state.persona_id_to_name_map.get(p_id)
                                if persona_name:
                                    persona_scores[persona_name] += feedback_score * CONTENT_FEEDBACK_WEIGHT
        logger.info(f"콘텐츠 피드백 기반 페르소나 점수 계산 완료. 현재 점수: {dict(persona_scores)}")
    else:
        logger.info(f"사용자 {user_id}의 콘텐츠 피드백이 없습니다.")

    if pbr_app_state.user_qa_answers_df is not None and not pbr_app_state.user_qa_answers_df.empty:
        user_answers = pbr_app_state.user_qa_answers_df[pbr_app_state.user_qa_answers_df['user_id'] == user_id]
        
        if pbr_app_state.options_df is None or pbr_app_state.options_df.empty:
            logger.warning("Options 데이터가 로드되지 않아 QA 답변 기반 페르소나 점수 계산을 건너뜝니다.")
        else:
            for _, answer_row in user_answers.iterrows():
                question_id = answer_row['question_id']
                option_id = answer_row['option_id']
                
                matched_option = pbr_app_state.options_df[
                    (pbr_app_state.options_df['question_id'] == question_id) &
                    (pbr_app_state.options_df['option_id'] == option_id)
                ]
                
                if not matched_option.empty:
                    answer_score_weight = matched_option['score'].iloc[0]
                    
                    relevant_persona_options = pbr_app_state.persona_options_df[
                        (pbr_app_state.persona_options_df['question_id'] == question_id) &
                        (pbr_app_state.persona_options_df['option_id'] == option_id)
                    ]
                    
                    for _, po_row in relevant_persona_options.iterrows():
                        persona_id_qa = po_row['persona_id']
                        persona_name_qa = pbr_app_state.persona_id_to_name_map.get(persona_id_qa)
                        if persona_name_qa:
                            persona_scores[persona_name_qa] += answer_score_weight

    all_persona_names = []
    if pbr_app_state.persona_details_map:
        for persona_detail in pbr_app_state.persona_details_map.values():
            persona_name = persona_detail.get('persona_name')
            if persona_name:
                all_persona_names.append(persona_name)
    else:
        logger.warning("persona_details_map이 로드되지 않았습니다. 모든 페르소나 이름을 가져올 수 없습니다.")

    for p_name in all_persona_names:
        if p_name not in persona_scores:
            persona_scores[p_name] = MIN_PERSONA_SCORE

    total_score_sum = sum(persona_scores.values())

    if total_score_sum > 0:
        for persona_name in persona_scores:
            persona_scores[persona_name] = (persona_scores[persona_name] / total_score_sum) * MAX_EXPECTED_PERSONA_SCORE
            persona_scores[persona_name] = max(persona_scores[persona_name], MIN_PERSONA_SCORE)
    else:
        for persona_name in all_persona_names:
            persona_scores[persona_name] = MIN_PERSONA_SCORE

    logger.info(f"사용자 {user_id}의 최종 페르소나 점수: {dict(persona_scores)}")
    return dict(persona_scores)


def get_hybrid_persona(
    user_id: int,
    unified_feedback: Dict[Tuple[int, str], float],
    db: Session
) -> Tuple[Optional[int], Optional[int], Optional[str], Optional[str], Dict[str, float]]:
    """
    사용자의 통합 피드백을 기반으로 메인 및 서브 페르소나를 결정합니다.
    "아기" 페르소나 로직을 적용합니다 (표시용).
    반환값: (main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores)
    """
    logger.info(f"사용자 {user_id}의 하이브리드 페르소나 결정 시작.")

    all_personas_scores = calculate_persona_scores_from_feedback(user_id, unified_feedback)
    logger.info(f"DEBUG: get_hybrid_persona 내부에서 all_personas_scores 확인: {all_personas_scores}")

    if not all_personas_scores:
        logger.warning(f"사용자 {user_id}에 대한 페르소나 점수를 계산할 수 없습니다. 기본값 반환.")
        return None, None, None, None, {}

    sorted_personas = sorted(all_personas_scores.items(), key=lambda item: (-item[1], pbr_app_state.persona_name_to_id_map.get(item[0], float('inf'))))

    main_persona_name = None
    sub_persona_name = None
    main_persona_id = None
    sub_persona_id = None
    main_persona_score = 0.0

    original_main_persona_name = None
    
    if sorted_personas:
        original_main_persona_name = sorted_personas[0][0]
        main_persona_score = sorted_personas[0][1]
        main_persona_id = pbr_app_state.persona_name_to_id_map.get(original_main_persona_name) 
        main_persona_name = original_main_persona_name

        if len(sorted_personas) > 1:
            sub_persona_name = sorted_personas[1][0]
            sub_persona_score = sorted_personas[1][1]
            sub_persona_id = pbr_app_state.persona_name_to_id_map.get(sub_persona_name)
        else: 
            sub_persona_name = None
            sub_persona_id = None
            sub_persona_score = 0.0

        is_baby_persona = False
        if sub_persona_name is not None:
            score1 = Decimal(str(main_persona_score))
            score2 = Decimal(str(sub_persona_score))
            totalScore = score1 + score2
            
            if totalScore.is_zero():
                main_percentage = Decimal(0)
                sub_percentage = Decimal(0)
                is_baby_persona = True 
            else:
                main_percentage = (score1 / totalScore * Decimal(100)).to_integral_value(rounding=ROUND_DOWN)
                sub_percentage = (score2 / totalScore * Decimal(100)).to_integral_value(rounding=ROUND_DOWN)

                if abs(main_percentage - sub_percentage) < 8:
                    is_baby_persona = True
        
        if is_baby_persona:
            logger.info(f"사용자 {user_id}: 메인 페르소나 '{original_main_persona_name}'({main_persona_score:.2f})와 서브 페르소나 '{sub_persona_name}'({sub_persona_score:.2f})의 비율 차이가 8%포인트 미만이므로 '아기' 페르소나로 분류합니다.")
            
            main_persona_name = f"아기 {original_main_persona_name}" 

        else:
            logger.info(f"사용자 {user_id}: 결정된 메인 페르소나: {main_persona_name} (ID: {main_persona_id if main_persona_id is not None else '없음'}), 서브 페르소나: {sub_persona_name if sub_persona_name else '없음'} (ID: {sub_persona_id if sub_persona_id is not None else '없음'})")
    else:
        logger.warning(f"사용자 {user_id}에 대해 정렬된 페르소나가 없습니다. 기본값 반환.")

    return main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores


def _get_general_popular_recommendations(user_id: int, num_recommendations: int, content_type_filter: Optional[str] = None, db: Session = None) -> List[RecommendedContent]:
    """
    모든 콘텐츠 중에서 가장 인기 있는 콘텐츠를 반환합니다.
    (장르나 페르소나에 관계없이 전체 데이터에서 인기 순으로 정렬)
    content_type_filter를 통해 특정 유형의 콘텐츠만 필터링할 수 있습니다.
    """
    logger.info(f"전체 인기 콘텐츠 {num_recommendations}개를 가져오는 중 (유형 필터: {content_type_filter}).")

    if pbr_app_state.contents_df is None or pbr_app_state.reactions_df is None:
        logger.warning("contents_df 또는 reactions_df가 로드되지 않았습니다. 인기 콘텐츠를 가져올 수 없습니다.")
        return []

    filtered_contents_df = pbr_app_state.contents_df
    if content_type_filter:
        filtered_contents_df = filtered_contents_df[filtered_contents_df['type'] == content_type_filter]
        if filtered_contents_df.empty:
            logger.warning(f"유형 '{content_type_filter}'에 해당하는 콘텐츠가 없습니다.")
            return []

    content_likes = pbr_app_state.reactions_df[pbr_app_state.reactions_df['reaction'] == ReactionType.LIKE] \
        .groupby(['content_id'])['content_id'].count().reset_index(name='like_count')
    
    content_dislikes = pbr_app_state.reactions_df[pbr_app_state.reactions_df['reaction'] == ReactionType.DISLIKE] \
        .groupby(['content_id'])['content_id'].count().reset_index(name='dislike_count')

    all_contents_with_popularity = pd.merge(filtered_contents_df, content_likes, on='content_id', how='left')
    all_contents_with_popularity = pd.merge(all_contents_with_popularity, content_dislikes, on='content_id', how='left')

    all_contents_with_popularity['like_count'] = all_contents_with_popularity['like_count'].fillna(0)
    all_contents_with_popularity['dislike_count'] = all_contents_with_popularity['dislike_count'].fillna(0)

    all_contents_with_popularity['popularity_score'] = all_contents_with_popularity['like_count'] - all_contents_with_popularity['dislike_count']

    user_idx = pbr_app_state.user_id_to_idx_map.get(user_id)
    if user_idx is not None and pbr_app_state.user_item_matrix is not None:
        viewed_content_indices = pbr_app_state.user_item_matrix[user_idx].nonzero()[1]
        viewed_content_ids = {pbr_app_state.idx_to_content_id_map[idx][0] for idx in viewed_content_indices}
        all_contents_with_popularity = all_contents_with_popularity[
            ~all_contents_with_popularity['content_id'].isin(viewed_content_ids)
        ]

    sorted_popular_contents = all_contents_with_popularity.sort_values(by='popularity_score', ascending=False)

    recommended_list = []
    current_count = 0
    for _, row in sorted_popular_contents.iterrows():
        if current_count >= num_recommendations:
            break

        recommended_list.append(_create_recommended_content(
            content_data=row.to_dict(),
            predicted_rating=row['popularity_score'],
            persona_genre_match=None
        ))
        current_count += 1
        
    logger.info(f"전체 인기 콘텐츠 {len(recommended_list)}개 생성 완료.")
    return recommended_list


def _get_popular_contents_for_persona(persona_id: int, num_recommendations: int) -> List[RecommendedContent]:
    """
    특정 페르소나와 관련된 인기 콘텐츠를 가져옵니다.
    페르소나에 매칭되는 장르가 없을 경우, 전체 인기 콘텐츠를 반환합니다.
    """
    logger.info(f"페르소나 ID {persona_id}를 위한 인기 기반 추천 생성 중...")

    if pbr_app_state.contents_df is None or pbr_app_state.reactions_df is None or pbr_app_state.persona_genres_df is None:
        logger.warning("필수 데이터프레임(contents_df, reactions_df, persona_genres_df)이 로드되지 않았습니다.")
        return []

    persona_matched_genres_df = pbr_app_state.persona_genres_df[
        (pbr_app_state.persona_genres_df['persona_id'] == persona_id)
    ]

    if persona_matched_genres_df.empty:
        logger.info(f"페르소나 ID {persona_id}에 매칭되는 장르가 없습니다. 전체 인기 콘텐츠로 폴백합니다.")
        return _get_general_popular_recommendations(user_id=None, num_recommendations=num_recommendations)

    persona_genre_ids = persona_matched_genres_df['genre_id'].unique().tolist()
    
    if not persona_genre_ids:
        logger.info(f"페르소나 ID {persona_id}에 연결된 장르 ID가 없습니다. 전체 인기 콘텐츠로 폴백합니다.")
        return _get_general_popular_recommendations(user_id=None, num_recommendations=num_recommendations)

    genre_id_to_name_map = pbr_app_state.genres_df.set_index('id')['name'].to_dict()
    persona_genre_names = [genre_id_to_name_map[gid] for gid in persona_genre_ids if gid in genre_id_to_name_map]

    if not persona_genre_names:
        logger.info(f"페르소나 ID {persona_id}에 연결된 유효한 장르 이름이 없습니다. 전체 인기 콘텐츠로 폴백합니다.")
        return _get_general_popular_recommendations(user_id=None, num_recommendations=num_recommendations)
    
    matched_contents = pbr_app_state.contents_df[
        pbr_app_state.contents_df['genres'].apply(
        lambda content_genres: bool(set(content_genres if content_genres is not None else []).intersection(persona_genre_names))
        )   
    ]

    persona_excluded_genre_names: List[str] = []
    persona_detail = pbr_app_state.persona_details_map.get(persona_id)
    if persona_detail and 'excluded_genres' in persona_detail:
        persona_excluded_genre_names = persona_detail['excluded_genres']
        logger.debug(f"페르소나 ID {persona_id}의 제외 장르: {persona_excluded_genre_names}")

    if persona_excluded_genre_names and not matched_contents.empty:
        initial_matched_count = matched_contents.shape[0]
        
        filtered_contents_without_excluded = matched_contents[
            ~matched_contents['genres'].apply(
                lambda content_genres: bool(set(content_genres if content_genres is not None else []).intersection(persona_excluded_genre_names))
            )
        ]
        matched_contents = filtered_contents_without_excluded
        if initial_matched_count != matched_contents.shape[0]:
            logger.info(f"페르소나 ID {persona_id}: 제외 장르 필터링 후 {initial_matched_count}개에서 {matched_contents.shape[0]}개로 줄어들었습니다.")

    if matched_contents.empty:
        logger.info(f"페르소나 ID {persona_id}의 장르와 매칭되는 콘텐츠가 없습니다. 전체 인기 콘텐츠로 폴백합니다.")
        return _get_general_popular_recommendations(user_id=None, num_recommendations=num_recommendations)

    content_likes = pbr_app_state.reactions_df[pbr_app_state.reactions_df['reaction'] == ReactionType.LIKE] \
        .groupby(['content_id'])['content_id'].count().reset_index(name='like_count')
    
    content_dislikes = pbr_app_state.reactions_df[pbr_app_state.reactions_df['reaction'] == ReactionType.DISLIKE] \
        .groupby(['content_id'])['content_id'].count().reset_index(name='dislike_count')

    persona_contents_with_popularity = pd.merge(matched_contents, content_likes, on='content_id', how='left')
    persona_contents_with_popularity = pd.merge(persona_contents_with_popularity, content_dislikes, on='content_id', how='left')

    persona_contents_with_popularity['like_count'] = persona_contents_with_popularity['like_count'].fillna(0)
    persona_contents_with_popularity['dislike_count'] = persona_contents_with_popularity['dislike_count'].fillna(0)
    persona_contents_with_popularity['popularity_score'] = persona_contents_with_popularity['like_count'] - persona_contents_with_popularity['dislike_count']

    top_popular_contents = persona_contents_with_popularity.sort_values(by='popularity_score', ascending=False)

    recommended_list = []
    current_count = 0
    for _, row in top_popular_contents.iterrows():
        if current_count >= num_recommendations:
            break

        content_info = _create_recommended_content(
            content_data=row.to_dict(), 
            predicted_rating=row['popularity_score'],
            persona_genre_match=True
        )
        
        if content_info.contentId is not None:
            recommended_list.append(content_info)
            current_count += 1
    
    logger.info(f"페르소나 ID {persona_id}를 위한 인기 기반 추천 {len(recommended_list)}개 생성 완료.")
    return recommended_list


def update_and_get_recommendations(
    user_id: int,
    db: Session,
    num_recommendations: int = RECOMMENDATION_COUNT,
    content_type_filter: Optional[str] = None) -> List[RecommendedContent]:
    """
    주어진 사용자 ID에 대한 추천 목록을 업데이트하고 반환합니다.
    """
    logger.info(f"사용자 {user_id}에 대한 추천 업데이트 및 조회 시작.")

    unified_feedback = generate_unified_user_feedback(user_id, db)
    main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores = \
        get_hybrid_persona(user_id, unified_feedback, db)

    if main_persona_id is None:
        logger.info(f"사용자 {user_id}의 메인 페르소나가 결정되지 않아 일반 인기 추천을 제공합니다.")
        return _get_general_popular_recommendations(user_id, num_recommendations)

    recommendations_for_persona: List[RecommendedContent] = _get_popular_contents_for_persona(
        main_persona_id, num_recommendations=RECOMMENDATION_COUNT
    )

    seen_content_keys = set()
    unique_recommendations: List[RecommendedContent] = []

    for rec in recommendations_for_persona:
        content_key = (rec.contentId, rec.type)
        if content_key not in seen_content_keys:
            seen_content_keys.add(content_key)
            unique_recommendations.append(rec)

    return unique_recommendations[:num_recommendations]

def get_persona_based_popular_fallback_recommendations(
    user_id: int,
    main_persona_id: int,
    num_recommendations: int, 
    db: Session,
    content_type_filter: Optional[str] = None
) -> List[RecommendedContent]:
    """
    페르소나 기반 인기 추천을 제공합니다.
    주어진 main_persona_id에 해당하는 장르를 기반으로 인기 콘텐츠를 필터링합니다.
    _get_popular_contents_for_persona 함수를 활용합니다.
    """
    logger.info(f"사용자 {user_id}를 위한 페르소나 {main_persona_id} 기반 인기 추천 생성 중 (요청 개수: {num_recommendations}, 유형 필터: {content_type_filter}).")

    initial_fetch_count = max(num_recommendations * 5, 50)
    raw_persona_popular_contents = _get_popular_contents_for_persona(main_persona_id, initial_fetch_count)
    logger.info(f"페르소나 ID {main_persona_id}를 위한 인기 기반 추천 초기 {len(raw_persona_popular_contents)}개 생성 완료 (요청: {initial_fetch_count}).")


    current_filtered_list = []

    if content_type_filter:
        for content in raw_persona_popular_contents:
            if content.type == content_type_filter:
                current_filtered_list.append(content)
        logger.info(f"콘텐츠 타입 '{content_type_filter}' 필터링 후 {len(current_filtered_list)}개 남음.")
    else:
        current_filtered_list = list(raw_persona_popular_contents)

    user_idx = pbr_app_state.user_id_to_idx_map.get(user_id)
    if user_idx is not None and pbr_app_state.user_item_matrix is not None:
        viewed_content_indices = pbr_app_state.user_item_matrix[user_idx].nonzero()[1]
        
        viewed_content_ids = {pbr_app_state.idx_to_content_id_map[idx] for idx in viewed_content_indices 
                              if idx in pbr_app_state.idx_to_content_id_map}

        after_viewed_filter = []
        for rec in current_filtered_list:
            if rec.contentId not in viewed_content_ids:
                after_viewed_filter.append(rec)
        current_filtered_list = after_viewed_filter
        logger.info(f"이미 본 콘텐츠 제외 후 {len(current_filtered_list)}개 남음.")
    else:
        logger.info("사용자-아이템 매트릭스 정보를 찾을 수 없거나 사용자 인덱스가 없습니다. 시청 기록 필터링 건너_get_popular_contents_for_persona.")

    final_recommendations = current_filtered_list[:num_recommendations]
    
    logger.info(f"사용자 {user_id}를 위한 페르소나 {main_persona_id} 기반 인기 추천 최종 {len(final_recommendations)}개 생성 완료 (요청 개수: {num_recommendations}).")
    return final_recommendations

def has_genre_match(content_genres: List[str], persona_genres: List[str]) -> Optional[bool]:
    if not content_genres or not persona_genres:
        return None
    return any(genre in persona_genres for genre in content_genres)


def recommend_contents_cf(
    user_id: int,
    num_recommendations: int,
    db: Session,
    content_type_filter: Optional[str] = None,
    persona_genres: Optional[List[str]] = None
) -> List[RecommendedContent]:
    """
    사용자 기반 협업 필터링(User-Based Collaborative Filtering)을 사용하여
    주어진 사용자에게 콘텐츠를 추천합니다.
    """
    logger.info(f"사용자 {user_id}에게 협업 필터링 기반 추천 생성 중 (유형 필터: {content_type_filter})...")
    
    if pbr_app_state.user_item_matrix is not None and \
    (pbr_app_state.user_similarity_df is None or pbr_app_state.user_similarity_df.shape[0] == 0):
        logger.info("user_similarity_df가 초기화되지 않아 새로 계산합니다.")
        _, user_sim_matrix_sparse = calculate_user_similarity(
            pbr_app_state.user_item_matrix,
            pbr_app_state.user_id_to_idx_map 
        )
        pbr_app_state.user_similarity_df = user_sim_matrix_sparse 

        if pbr_app_state.user_similarity_df is None or pbr_app_state.user_similarity_df.shape[0] == 0:
            logger.warning("user_similarity_df 계산에 실패했거나 비어 있습니다. CF 추천을 생성할 수 없습니다.")
            return []

    if pbr_app_state.user_item_matrix is None or \
       pbr_app_state.user_similarity_df is None or \
       pbr_app_state.user_id_to_idx_map is None or \
       pbr_app_state.idx_to_content_id_map is None or \
       pbr_app_state.contents_df is None:
        logger.warning("필수 데이터(user_item_matrix, user_similarity_df, user/content mapping, contents_df)가 초기화되지 않았습니다.")
        return []

    if user_id not in pbr_app_state.user_id_to_idx_map:
        logger.warning(f"사용자 ID {user_id}를 매핑에서 찾을 수 없습니다. 추천을 생성할 수 없습니다.")
        return []

    user_idx = pbr_app_state.user_id_to_idx_map[user_id]
    num_users = pbr_app_state.user_item_matrix.shape[0]
    num_contents = pbr_app_state.user_item_matrix.shape[1]

    user_similarities_row = pbr_app_state.user_similarity_df.getrow(user_idx)

    similar_users_info = []
    for other_user_idx, similarity_score in zip(user_similarities_row.indices, user_similarities_row.data):
        if other_user_idx == user_idx: 
            continue
        if similarity_score > CF_SIMILARITY_THRESHOLD: 
            similar_users_info.append((other_user_idx, similarity_score))

    similar_users_info.sort(key=lambda x: x[1], reverse=True)

    predicted_ratings: Dict[Tuple[int, str], float] = defaultdict(float)
    similarity_sums: Dict[Tuple[int, str], float] = defaultdict(float)

    user_rated_content_ids: Set[Tuple[int, str]] = set()
    for c_idx in pbr_app_state.user_item_matrix[user_idx].nonzero()[1]:
        original_content_id, content_type_from_matrix = pbr_app_state.idx_to_content_id_map[c_idx]
        user_rated_content_ids.add((original_content_id, content_type_from_matrix))


    for other_user_idx, similarity in similar_users_info:
        other_user_ratings_indices = pbr_app_state.user_item_matrix[other_user_idx].nonzero()[1]
        for content_matrix_idx in other_user_ratings_indices:
            original_content_id, content_type_from_matrix = pbr_app_state.idx_to_content_id_map[content_matrix_idx]

            if (original_content_id, content_type_from_matrix) in user_rated_content_ids:
                continue

            if content_type_filter and content_type_filter != content_type_from_matrix:
                continue

            rating = pbr_app_state.user_item_matrix[other_user_idx, content_matrix_idx]

            predicted_ratings[(original_content_id, content_type_from_matrix)] += rating * similarity 
            similarity_sums[(original_content_id, content_type_from_matrix)] += abs(similarity)

    final_predicted_ratings: Dict[Tuple[int, str], float] = {} 
    for (content_id, content_type), total_score in predicted_ratings.items():
        if similarity_sums[(content_id, content_type)] > 0:
            final_predicted_ratings[(content_id, content_type)] = total_score / similarity_sums[(content_id, content_type)] 
        else:
            final_predicted_ratings[(content_id, content_type)] = 0 

    recommended_content_items = sorted(final_predicted_ratings.items(), key=lambda item: item[1], reverse=True)

    recommendations: List[RecommendedContent] = []
    seen_content_items: Set[Tuple[int, str]] = set()

    for (content_id, content_type), predicted_rating in recommended_content_items: 
        if (content_id, content_type) not in seen_content_items: 
            content_row_df = pbr_app_state.contents_df[
                (pbr_app_state.contents_df['content_id'] == content_id) &
                (pbr_app_state.contents_df['type'] == content_type)
            ]

            if content_row_df.empty:
                logger.warning(f"CF 추천: content_id {content_id}, type {content_type}의 정보를 pbr_app_state.contents_df에서 찾을 수 없습니다. 건너뜜.")
                continue 

            content_row = content_row_df.iloc[0].to_dict()

            content_info = _create_recommended_content(
                content_data=content_row,
                predicted_rating=predicted_rating
            )
            
            if content_info.contentId is not None:
                recommendations.append(content_info)
                seen_content_items.add((content_id, content_type))
        
        if len(recommendations) >= num_recommendations:
            break

    logger.info(f"사용자 {user_id}에게 {len(recommendations)}개의 협업 필터링 추천 생성 완료.")
    return recommendations

def _get_user_interacted_content_tuples(user_id: int) -> Set[Tuple[int, str]]:
    """
    주어진 사용자가 이미 반응하거나 리뷰한 모든 콘텐츠의 (content_id, content_type) 튜플 집합을 반환합니다.
    """
    user_reactions_df = pbr_app_state.reactions_df[pbr_app_state.reactions_df['user_id'] == user_id]
    reacted_content_tuples = set(zip(user_reactions_df['content_id'], user_reactions_df['type']))

    user_reviews_df = pbr_app_state.reviews_df[pbr_app_state.reviews_df['user_id'] == user_id]
    reviewed_content_tuples = set(zip(user_reviews_df['content_id'], user_reviews_df['type']))

    return reacted_content_tuples.union(reviewed_content_tuples)