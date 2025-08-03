import logging
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from fastapi import APIRouter, HTTPException, Depends, status, Query
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional, Set
from datetime import datetime
from collections import defaultdict

from sqlalchemy.orm import Session
from sqlalchemy import func, text

# 내부 모듈 임포트
from .state import pbr_app_state
from .config import (
    LIKE_WEIGHT, DISLIKE_WEIGHT, STAR_RATING_WEIGHTS,
    PERSONA_SIMILARITY_THRESHOLD, CONTENT_SIMILARITY_THRESHOLD,
    TOP_N_CONTENTS_FOR_POPULAR_PER_PERSONA,
    RECOMMENDATION_CACHE_EXPIRATION_SECONDS,
    CF_SIMILARITY_THRESHOLD,
    QA_WEIGHT, CONTENT_FEEDBACK_WEIGHT, MIN_FEEDBACK_FOR_PERSONA_DETERMINATION,
    MIN_POSITIVE_RATINGS_FOR_GROWTH, PENALTY_FOR_NO_PERSONA_MATCH,
    MIN_VOTE_COUNT_FOR_POPULARITY, QA_INITIAL_RECOMMENDATION_THRESHOLD,
    MAX_EXPECTED_PERSONA_SCORE, MIN_PERSONA_SCORE, BABY_PERSONA_THRESHOLD,
    POSITIVE_FEEDBACK_THRESHOLD, RECOMMENDATION_COUNT
)
from .schemas import RecommendedContent
from models import UserPersona, ContentReaction, Review, Persona, ReactionType # Persona 모델 임포트 필요

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _create_recommended_content(
    content_data: Dict, # content_id 대신 필터링된 콘텐츠 데이터를 받습니다.
    predicted_rating: Optional[float] = None,
    persona_genre_match: Optional[float] = None
) -> RecommendedContent:
    """
    주어진 콘텐츠 데이터 딕셔너리를 기반으로 RecommendedContent 객체를 생성합니다.
    (이미 필터링된 DataFrame의 row에서 온 데이터를 사용합니다.)
    """
    # content_data는 이미 필터링된 DataFrame의 row에서 온 것이므로, 추가 조회나 None 체크 불필요
    
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
            genres_list = [genres_data.strip()] # 단일 장르


    return RecommendedContent(
        contentId=int(content_data.get('content_id')), # content_id가 int인지 확인
        title=content_data.get('title', ''),
        genres=genres_list, # 파싱된 장르 리스트 사용
        type=content_data.get('type', ''), # 이미 필터링된 정확한 type 정보 사용
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

    # LIL(List of Lists) 형식의 희소 행렬을 생성합니다. 행렬 원소 추가에 효율적입니다.
    user_item_matrix = lil_matrix((num_users, num_contents))

    # Reactions 데이터 처리
    for _, row in reactions_df.iterrows():
        user_id = row['user_id']
        content_id_orig = row['content_id']
        content_type = row['type']
        reaction_type = row['reaction_type']

        user_idx = user_id_to_idx_map.get(user_id)
        content_tuple = (content_id_orig, content_type)
        content_idx = content_id_to_idx_map.get(content_tuple)

        if user_idx is not None and content_idx is not None:
            # 'liked'에 가중치 1, 'disliked'에 가중치 -1 부여 (예시)
            # 필요에 따라 가중치 조정 또는 다른 방식으로 점수화 가능
            score = 0
            if reaction_type == 'liked':
                score = 1
            elif reaction_type == 'disliked':
                score = -1
            
            # 이미 값이 있으면 합산 (예: 같은 콘텐츠에 대해 반응과 리뷰가 모두 있는 경우)
            user_item_matrix[user_idx, content_idx] += score

    # Reviews 데이터 처리
    for _, row in reviews_df.iterrows():
        user_id = row['user_id']
        content_id_orig = row['content_id'] # reviews_df에서는 'content_id' 컬럼 사용
        content_type = row['content_type'] # reviews_df에서는 'content_type' 컬럼 사용
        rating = row['rating']

        user_idx = user_id_to_idx_map.get(user_id)
        content_tuple = (content_id_orig, content_type)
        content_idx = content_id_to_idx_map.get(content_tuple)

        if user_idx is not None and content_idx is not None:
            # 리뷰 평점을 그대로 사용하거나 스케일링하여 사용
            user_item_matrix[user_idx, content_idx] += rating # 또는 rating / MAX_RATING 등으로 정규화

    # CSR(Compressed Sparse Row) 형식으로 변환하여 효율적인 계산을 준비
    user_item_matrix_csr = user_item_matrix.tocsr()
    logger.info(f"사용자-콘텐츠 희소 행렬 계산 완료. Shape: {user_item_matrix_csr.shape}")
    return user_item_matrix_csr

def calculate_and_store_user_personas(
    db: Session,
    user_id: int,
    calculated_persona_scores: Dict[str, float] # 타입 힌트를 str로 변경했습니다 (선택 사항)
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

    for persona_id, new_score in calculated_persona_scores.items(): # 이제 persona_id에 올바른 숫자 ID가 들어옵니다.
        # persona_name은 로깅이나 다른 목적에 필요한 경우에만 다시 매핑합니다.
        persona_name = pbr_app_state.persona_id_to_name_map.get(persona_id) 
        
        # 이제 올바른 정수형 persona_id를 사용하여 쿼리합니다.
        persona = db.query(Persona).filter(Persona.persona_id == persona_id).first()
        if not persona:
            # 이 경고는 이제 Persona 마스터 데이터가 실제로 없는 경우에만 발생해야 합니다.
            persona_name_for_log = pbr_app_state.persona_id_to_name_map.get(persona_id, f"Unknown ID {persona_id}")
            logger.warning(f"DB Persona 테이블에서 페르소나 ID {persona_id} (이름: '{persona_name_for_log}')를 찾을 수 없습니다. 해당 페르소나 점수는 저장되지 않습니다.")
            continue

        persona_ids_to_keep.add(persona_id)
        if persona_id in existing_personas_map:
            # 기존 페르소나가 있다면 업데이트
            user_persona = existing_personas_map[persona_id]
            score_changed = (user_persona.score != new_score)

            if score_changed:
                user_persona.score = new_score
                logger.debug(f"사용자 {user_id}, 페르소나 '{persona_name}' (ID: {persona_id}): 점수 업데이트됨 ({user_persona.score} -> {new_score}).")
            else:
                logger.debug(f"사용자 {user_id}, 페르소나 '{persona_name}' (ID: {persona_id}): 점수 변화 없음 ({new_score}).")

            user_persona.updated_at = current_utc_time

        else:
            # 새로운 페르소나라면 삽입
            new_user_persona = UserPersona(
                user_id=user_id,
                persona_id=persona_id, # 여기에도 올바른 persona_id 사용
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
    except Exception as e:
        logger.error(f"DB 변경사항 플러시 중 오류 발생: {e}", exc_info=True)
        raise

    logger.info(f"사용자 {user_id}의 페르소나 점수 DB 저장 완료 (Upsert 방식).")


def calculate_user_similarity(user_item_matrix: csr_matrix, user_idx_to_id_map: Dict[int, int]) -> Tuple[pd.DataFrame, csr_matrix]:
    logger.info("사용자 유사도 계산 시작...")
    if user_item_matrix.shape[0] == 0:
        logger.warning("user_item_matrix가 비어 있어 사용자 유사도를 계산할 수 없습니다.")
        # 빈 데이터프레임과 빈 희소 행렬 반환
        return pd.DataFrame(columns=['user1_idx', 'user2_idx', 'similarity_score']), csr_matrix((0,0))

    # 사용자-사용자 코사인 유사도 계산
    user_similarity_matrix = cosine_similarity(user_item_matrix)
    logger.info(f"user_similarity_matrix shape: {user_similarity_matrix.shape}")

    # 유사도 결과를 DataFrame으로 변환
    similarity_data = []
    num_users = user_item_matrix.shape[0]
    for i in range(num_users):
        for j in range(i + 1, num_users): # 중복 계산 방지 및 자기 자신 제외
            similarity_score = user_similarity_matrix[i, j]
            if similarity_score > 0: # 0보다 큰 유사도만 기록 (조정 가능)
                similarity_data.append({
                    'user1_idx': i,
                    'user2_idx': j,
                    'similarity_score': similarity_score
                })

    final_similarity_df = pd.DataFrame(similarity_data)
    logger.info(f"사용자 유사도 계산 완료. 유사한 사용자 쌍: {len(final_similarity_df)}")
    
    # user_similarity_matrix도 csr_matrix 형태로 반환
    user_similarity_matrix_sparse = csr_matrix(user_similarity_matrix)
    
    return final_similarity_df, user_similarity_matrix_sparse # DataFrame과 Sparse Matrix 모두 반환


def calculate_content_similarity_sparse(user_item_matrix: csr_matrix) -> csr_matrix:
    logger.info("콘텐츠 유사도 계산 시작...")
    if user_item_matrix.shape[1] == 0: # 콘텐츠 수가 0인지 확인
        logger.warning("user_item_matrix에 콘텐츠가 없어 콘텐츠 유사도를 계산할 수 없습니다.")
        return csr_matrix((0,0)) # 빈 희소 행렬 반환

    # 콘텐츠-콘텐츠 코사인 유사도 계산 (사용자-아이템 행렬의 전치에 대해 계산)
    content_similarity_matrix = cosine_similarity(user_item_matrix.T)
    logger.info(f"콘텐츠 유사도 계산 완료. Shape: {content_similarity_matrix.shape}")
    return csr_matrix(content_similarity_matrix) # 희소 행렬로 반환


def calculate_persona_similarity() -> pd.DataFrame:
    """
    페르소나 간의 유사도를 계산합니다.
    페르소나-장르 매핑을 기반으로 코사인 유사도를 사용합니다.
    pbr_app_state.persona_df에는 'persona_id'와 'name' 컬럼이 있고,
    pbr_app_state.genres_df에는 'id'와 'name' 컬럼이 있다고 가정합니다.
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

    # 페르소나-장르 데이터와 장르 데이터를 병합
    persona_genre_merged = persona_genres_df.merge(
        genres_df, left_on='genre_id', right_on='id', suffixes=('_persona', '_genre')
    )

    # 피벗 테이블을 사용하여 페르소나-장르 원-핫 인코딩 매트릭스 생성
    # 각 페르소나가 어떤 장르를 가지고 있는지 0 또는 1로 표시
    persona_genre_matrix = persona_genre_merged.pivot_table(
        index='persona_id', columns='name', aggfunc='size', fill_value=0
    ).fillna(0)

    if persona_genre_matrix.empty:
        logger.warning("페르소나-장르 매트릭스가 비어 있습니다. 페르소나 유사도를 계산할 수 없습니다.")
        return pd.DataFrame(columns=['persona_id_1', 'persona_id_2', 'similarity_score'])

    # 코사인 유사도 계산
    persona_similarity = cosine_similarity(persona_genre_matrix)
    persona_similarity_df = pd.DataFrame(persona_similarity,
                                         index=persona_genre_matrix.index,
                                         columns=persona_genre_matrix.index)

    # 결과 DataFrame 생성
    similarities = []
    for i in range(persona_similarity_df.shape[0]):
        for j in range(i + 1, persona_similarity_df.shape[1]): # 중복 및 자기 자신 제외
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
    (이전 data_loader에서 persona_details_map을 만들 때 키워드를 포함시켰으므로,
    여기서는 해당 정보를 활용하거나 필요에 따라 생성 로직을 변경합니다.)
    """
    logger.info("페르소나 키워드 생성 시작...")
    
    persona_keyword_map: Dict[int, List[str]] = {}
    
    # pbr_app_state에 persona_details_map이 이미 채워져 있다고 가정
    # 이 맵에서 'keywords' 정보를 가져와 사용합니다.
    if pbr_app_state.persona_details_map:
        for persona_id, details in pbr_app_state.persona_details_map.items():
            if 'keywords' in details:
                persona_keyword_map[persona_id] = details['keywords']
    else:
        logger.warning("pbr_app_state.persona_details_map이 비어 있습니다. 키워드를 생성할 수 없습니다.")
        # 또는 persona_df와 genre_df, persona_genres_df를 사용하여 직접 생성하는 로직 추가
        # (이전에 data_loader에서 이미 처리했으므로 여기서는 단순히 가져오는 것을 가정합니다.)
        
    logger.info("페르소나 키워드 생성 완료.")
    return persona_keyword_map


def update_user_persona_scores(
    user_id: int,
    db: Session
):
    logger.info(f"사용자 {user_id}의 페르소나 점수 업데이트 시작.")

    # 1. 사용자 통합 피드백 생성 (최신 reactions, reviews, qa_answers 포함)
    unified_feedback = generate_unified_user_feedback(user_id, db)

    # 2. 통합 피드백 기반 페르소나 점수 계산
    calculated_persona_scores_by_name = calculate_persona_scores_from_feedback(user_id, unified_feedback)

    logger.info(f"DEBUG: calculate_persona_scores_by_name 결과: {calculated_persona_scores_by_name}")
    
    # 3. 페르소나 이름 -> ID 매핑을 사용하여 ID 기반 딕셔너리 생성
    calculated_persona_scores_by_id = {}
    for name, score in calculated_persona_scores_by_name.items():
        original_name = name # 원본 이름 로깅을 위해 저장
        normalized_name = name.strip() # 추가: 이름의 양쪽 공백 제거

        # 추가: 정규화된 이름과 매핑 존재 여부 로깅
        if normalized_name in pbr_app_state.persona_name_to_id_map:
            persona_id = pbr_app_state.persona_name_to_id_map[normalized_name]
            calculated_persona_scores_by_id[persona_id] = score
            logger.info(f"DEBUG: 페르소나 '{original_name}' (정규화: '{normalized_name}') -> ID {persona_id}로 성공적으로 매핑되었습니다.")
        else:
            logger.warning(f"경고: 페르소나 이름 '{original_name}' (정규화: '{normalized_name}')이(가) pbr_app_state.persona_name_to_id_map에서 찾아지지 않았습니다.")
            # 추가: pbr_app_state.persona_name_to_id_map의 현재 키 목록을 출력하여 비교
            logger.warning(f"DEBUG: pbr_app_state.persona_name_to_id_map의 현재 키: {list(pbr_app_state.persona_name_to_id_map.keys())}")

    # 4. DB 및 pbr_app_state에 페르소나 점수 저장 (기존 로직)
    # 이제 이 함수가 calculate_persona_scores_from_feedback에서 계산된 점수를 사용합니다.
    try:
        calculate_and_store_user_personas(db, user_id, calculated_persona_scores_by_id)
        logger.info(f"사용자 {user_id}의 페르소나 점수 DB 업데이트 완료.")
    except Exception as e:
        logger.error(f"사용자 {user_id}의 페르소나 점수 DB 저장 중 오류 발생: {e}", exc_info=True)
        # 여기서 예외를 다시 발생시켜 상위 함수에서 처리하도록 할 수 있습니다.
        # raise

    # --- 추가된 부분: 메모리 내 pbr_app_state 데이터 갱신 ---
    logger.info(f"메모리 내 pbr_app_state.all_user_personas_df 및 user_persona_scores_map 갱신 시도.")
    try:
        # DB에서 최신 UserPersonas 데이터 다시 로드
        result = db.execute(text("SELECT * FROM user_personas"))
        all_user_personas_df_updated = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.all_user_personas_df = all_user_personas_df_updated
        
        # user_persona_scores_map 재구축
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
    # --- 추가된 부분 끝 ---

    logger.info(f"사용자 {user_id}의 페르소나 점수 업데이트 처리 완료.")


def generate_unified_user_feedback(user_id: int, db: Session) -> Dict[Tuple[int, str], float]:
    """
    사용자의 좋아요/싫어요 및 리뷰 데이터를 통합하여 각 콘텐츠에 대한 피드백 점수를 생성합니다.
    이 함수는 User QA Answers (pbr_app_state.user_qa_answers_df)에 대한 가중치도 고려할 수 있습니다.
    ContentReaction 및 Review 객체에는 'content_id' 필드가 있다고 가정합니다.
    """
    logger.info(f"사용자 {user_id}의 통합 피드백 생성 중...")
    
    # unified_feedback의 초기화 타입을 반환 타입과 일치시킵니다.
    unified_feedback: Dict[Tuple[int, str], float] = defaultdict(float) 

    # 1. 좋아요/싫어요 반응 처리
    reactions = db.query(ContentReaction).filter(ContentReaction.user_id == user_id).all()
    for reaction in reactions:
        key = (reaction.content_id, reaction.type) # 이미 튜플 키를 사용 중
        if reaction.reaction == ReactionType.LIKE:
            unified_feedback[key] += LIKE_WEIGHT
        elif reaction.reaction == ReactionType.DISLIKE:
            unified_feedback[key] += DISLIKE_WEIGHT
    logger.info(f"ContentReaction {len(reactions)}개 처리 완료.")

    # 2. 리뷰(별점) 처리
    reviews = db.query(Review).filter(Review.user_id == user_id).all()
    for review in reviews:
        key = (review.content_id, review.type) # 이미 튜플 키를 사용 중
        score = STAR_RATING_WEIGHTS.get(review.score, 0.0)
        unified_feedback[key] += score
    logger.info(f"Review {len(reviews)}개 처리 완료.")

    # # 3. QA 답변 데이터 처리 (pbr_app_state.user_qa_answers_df 사용)
    # if pbr_app_state.user_qa_answers_df is not None and not pbr_app_state.user_qa_answers_df.empty:
    #     user_qa_df = pbr_app_state.user_qa_answers_df[pbr_app_state.user_qa_answers_df['user_id'] == user_id]
        
    #     # contents_df가 로드되어 있어야 content_type을 찾을 수 있습니다.
    #     if pbr_app_state.contents_df.empty:
    #         logger.warning("QA 답변 처리를 위해 pbr_app_state.contents_df가 비어 있습니다. QA 점수가 반영되지 않을 수 있습니다.")
        
    #     for _, row in user_qa_df.iterrows():
    #         qa_content_id = row.get('content_id')
    #         qa_score = row.get('score')
            
    #         if qa_content_id is not None and qa_score is not None:
    #             qa_content_type = None
    #             # QA 답변의 content_id에 해당하는 content_type을 contents_df에서 찾습니다.
    #             # 참고: content_id가 여러 type에 존재할 가능성은 낮지만, 정확성을 위해 필터링합니다.
    #             content_match = pbr_app_state.contents_df[pbr_app_state.contents_df['content_id'] == qa_content_id]
                
    #             if not content_match.empty:
    #                 # 일반적으로 하나의 content_id는 하나의 type을 가질 것입니다.
    #                 # 만약 여러 타입이 있다면, 첫 번째 타입을 사용하거나 명확한 로직이 필요합니다.
    #                 qa_content_type = content_match['type'].iloc[0]
    #             else:
    #                 logger.warning(f"QA 답변 콘텐츠 ID {qa_content_id}에 대한 content_type을 찾을 수 없습니다. QA 점수를 반영하지 않습니다.")
    #                 continue # content_type을 찾지 못했으므로 다음 QA 답변으로 넘어감
                
    #             if qa_content_type: # content_type을 성공적으로 찾았다면
    #                 # 튜플 키를 사용하여 unified_feedback에 점수를 추가합니다.
    #                 key = (qa_content_id, qa_content_type)
    #                 unified_feedback[key] += qa_score * QA_WEIGHT
                
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
    
    # 먼저 장르 ID와 이름 매핑 생성
    genre_id_to_name = pbr_app_state.genres_df.set_index('id')['name'].to_dict()

    # persona_genres_df를 순회하며 매핑 생성
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

    # 페르소나 이름으로 점수를 저장할 딕셔너리
    persona_scores: Dict[str, float] = defaultdict(float) 

    # 1. 콘텐츠 기반 피드백 처리 (긍정적인 반응 장르 수집)
    user_positive_genres: Set[str] = set()
    if unified_feedback: # 통합 피드백이 비어있지 않은 경우에만 처리
        logger.info(f"사용자 {user_id}의 콘텐츠 피드백 기반 페르소나 점수 계산 시작.")
        
        # contents_df에 'content_id'가 필요하다면, 이 부분은 data_loader에서 미리 처리하는 것이 좋습니다.
        # 하지만 여기서는 방어적으로 다시 체크합니다.
        if 'content_id' not in pbr_app_state.contents_df.columns or \
           not pd.api.types.is_numeric_dtype(pbr_app_state.contents_df['content_id']):
            pbr_app_state.contents_df['content_id'] = pd.to_numeric(
                pbr_app_state.contents_df['content_id'], errors='coerce'
            )
            logger.warning("pbr_app_state.contents_df['content_id']를 'content_id'로 변환했습니다.")

        for (content_id, content_type), feedback_score in unified_feedback.items():
            # 긍정적인 피드백 또는 모든 피드백을 고려할지 결정 (현재 코드는 >0만 고려)
            if feedback_score > 0: 
                # 정수형 content_id_orig와 type을 사용하여 필터링
                content_row = pbr_app_state.contents_df[
                    (pbr_app_state.contents_df['content_id'] == content_id) & # 'content_id' 사용
                    (pbr_app_state.contents_df['type'] == content_type)
                ]
                
                if not content_row.empty: # content_row가 비어있지 않은 경우에만 접근
                    genres_data = content_row['genres'].iloc[0]
                    
                    # FIX: 장르 문자열을 '|' 기준으로 분리하고 공백 제거
                    if isinstance(genres_data, str):
                        genres_list = [g.strip() for g in genres_data.split('|') if g.strip()] # <-- FIX
                    elif isinstance(genres_data, list): # 이미 리스트인 경우를 위한 방어 코드 (data_loader에 따라 다름)
                        genres_list = [g.strip() for g in genres_data if isinstance(g, str) and g.strip()]
                    else:
                        genres_list = []
                        logger.warning(f"콘텐츠 {content_id}의 장르 데이터 형식이 예상과 다릅니다: {type(genres_data)}. 스킵합니다.")

                    if genres_list: # 장르 리스트가 비어있지 않은 경우에만 처리
                        user_positive_genres.update(genres_list)

                    # 콘텐츠 피드백 기반 페르소나 점수 추가 (기존 로직 유지)
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


    # 각 페르소나의 키워드(장르)와 사용자 긍정 장르를 비교하여 점수 계산
    # 이 부분은 QA 답변 처리와 별개로, 'user_positive_genres'를 직접 활용하여 점수를 초기화하거나 추가할 수 있습니다.
    # 기존 코드에서 이 로직이 'user_positive_genres'를 기반으로 이미 점수를 매기고 있으므로, 이 부분은 통합되었습니다.

    # 2. QA 답변 기반 피드백 처리
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
                        persona_name_qa = pbr_app_state.persona_id_to_name_map.get(persona_id_qa) # 페르소나 이름으로 변환
                        if persona_name_qa:
                            persona_scores[persona_name_qa] += answer_score_weight


    # 3. 점수 정규화 및 최소 점수 적용
    # 모든 페르소나 이름을 가져와서 점수가 없는 페르소나에도 최소 점수를 부여
    all_persona_names = []
    if pbr_app_state.persona_details_map:
        for persona_detail in pbr_app_state.persona_details_map.values():
            persona_name = persona_detail.get('persona_name')
            if persona_name:
                all_persona_names.append(persona_name)
    else:
        logger.warning("persona_details_map이 로드되지 않았습니다. 모든 페르소나 이름을 가져올 수 없습니다.")

    # 모든 페르소나 이름에 대해 최소 점수 적용
    for p_name in all_persona_names:
        if p_name not in persona_scores:
            persona_scores[p_name] = MIN_PERSONA_SCORE

    total_score_sum = sum(persona_scores.values())

    if total_score_sum > 0:
        for persona_name in persona_scores:
            # 점수 정규화
            persona_scores[persona_name] = (persona_scores[persona_name] / total_score_sum) * MAX_EXPECTED_PERSONA_SCORE
            # 최소 점수 보장
            persona_scores[persona_name] = max(persona_scores[persona_name], MIN_PERSONA_SCORE)
    else:
        # 모든 점수가 0이거나 피드백이 없을 경우, 모든 페르소나에 최소 점수 부여
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

    # 점수를 내림차순으로 정렬
    sorted_personas = sorted(all_personas_scores.items(), key=lambda item: item[1], reverse=True)

    main_persona_name = None
    sub_persona_name = None
    main_persona_id = None
    sub_persona_id = None
    main_persona_score = 0.0

    original_main_persona_name = None # '아기' 접두사를 붙이기 전의 원래 메인 페르소나 이름
    
    if sorted_personas:
        # 항상 가장 높은 점수의 페르소나가 메인 페르소나의 기반이 됩니다.
        original_main_persona_name = sorted_personas[0][0]
        main_persona_score = sorted_personas[0][1]
        main_persona_id = pbr_app_state.persona_name_to_id_map.get(original_main_persona_name) # <-- 실제 추천에 사용될 ID
        main_persona_name = original_main_persona_name # API 응답에 사용될 이름 (아직 '아기' 접두사 없음)

        # 두 번째로 높은 점수의 페르소나가 서브 페르소나
        if len(sorted_personas) > 1:
            sub_persona_name = sorted_personas[1][0]
            sub_persona_score = sorted_personas[1][1]
            sub_persona_id = pbr_app_state.persona_name_to_id_map.get(sub_persona_name)
        else: 
            sub_persona_name = None
            sub_persona_id = None
            sub_persona_score = 0.0 # 서브 페르소라 점수 초기화 (계산에 사용)

        # "아기" 페르소나 로직 (표시용):
        # 메인과 서브 페르소나가 모두 존재하고 점수 차이가 10 미만일 때 "아기" 접두사 적용
        if sub_persona_name is not None and (main_persona_score - sub_persona_score) < 10:
            logger.info(f"사용자 {user_id}: 메인 페르소나 '{original_main_persona_name}'({main_persona_score:.2f})와 서브 페르소나 '{sub_persona_name}'({sub_persona_score:.2f})의 점수 차이가 10 미만이므로 '아기' 페르소나로 분류합니다 (표시용).")
            
            # API 응답에 사용될 main_persona_name에 "아기" 접두사 추가
            main_persona_name = f"아기 {original_main_persona_name}" 
            
            # 중요: main_persona_id는 계속 원래 페르소나의 ID를 유지합니다. (추천 로직 영향 없음)
            # all_personas_scores의 값은 변경하지 않고 원본 점수를 유지합니다.

        else:
            logger.info(f"사용자 {user_id}: 결정된 메인 페르소나: {main_persona_name} (ID: {main_persona_id if main_persona_id is not None else '없음'}), 서브 페르소나: {sub_persona_name if sub_persona_name else '없음'} (ID: {sub_persona_id if sub_persona_id is not None else '없음'})")
    else:
        logger.warning(f"사용자 {user_id}에 대해 정렬된 페르소나가 없습니다. 기본값 반환.")

    # 반환되는 main_persona_id는 추천 로직에서 사용되며,
    # '아기' 페르소나의 경우에도 실제 페르소나 ID를 가집니다.
    # main_persona_name은 '아기' 접두사가 붙을 수 있습니다.
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

    # 콘텐츠 타입 필터링
    filtered_contents_df = pbr_app_state.contents_df
    if content_type_filter:
        filtered_contents_df = filtered_contents_df[filtered_contents_df['type'] == content_type_filter]
        if filtered_contents_df.empty:
            logger.warning(f"유형 '{content_type_filter}'에 해당하는 콘텐츠가 없습니다.")
            return []

    # 각 콘텐츠별 좋아요/싫어요 개수 집계
    content_likes = pbr_app_state.reactions_df[pbr_app_state.reactions_df['reaction'] == ReactionType.LIKE] \
        .groupby(['content_id'])['content_id'].count().reset_index(name='like_count')
    
    content_dislikes = pbr_app_state.reactions_df[pbr_app_state.reactions_df['reaction'] == ReactionType.DISLIKE] \
        .groupby(['content_id'])['content_id'].count().reset_index(name='dislike_count')

    # 필터링된 콘텐츠 데이터와 좋아요/싫어요 데이터 병합
    all_contents_with_popularity = pd.merge(filtered_contents_df, content_likes, on='content_id', how='left')
    all_contents_with_popularity = pd.merge(all_contents_with_popularity, content_dislikes, on='content_id', how='left')

    # NaN 값을 0으로 채우기
    all_contents_with_popularity['like_count'] = all_contents_with_popularity['like_count'].fillna(0)
    all_contents_with_popularity['dislike_count'] = all_contents_with_popularity['dislike_count'].fillna(0)

    # 인기 점수 계산
    all_contents_with_popularity['popularity_score'] = all_contents_with_popularity['like_count'] - all_contents_with_popularity['dislike_count']

    # 이미 본 콘텐츠 제외 (여기서는 user_id를 받으므로, 실제 유저의 시청 기록을 제외하는 로직 추가)
    user_idx = pbr_app_state.user_id_to_idx_map.get(user_id)
    if user_idx is not None and pbr_app_state.user_item_matrix is not None:
        viewed_content_indices = pbr_app_state.user_item_matrix[user_idx].nonzero()[1]
        # map의 value가 튜플 (content_id, type)이므로, 첫 번째 요소인 content_id만 추출합니다.
        viewed_content_ids = {pbr_app_state.idx_to_content_id_map[idx][0] for idx in viewed_content_indices}
        all_contents_with_popularity = all_contents_with_popularity[
            ~all_contents_with_popularity['content_id'].isin(viewed_content_ids)
        ]

    # 인기 점수 기준으로 정렬 및 상위 N개 콘텐츠 선택 (필터링 후 개수가 줄어들 수 있음)
    # 이 부분에서 .head(num_recommendations)를 하면, 제외된 후 개수가 적어질 수 있습니다.
    # 따라서 정렬만 하고, 아래 루프에서 개수를 제어하는 것이 좋습니다.
    sorted_popular_contents = all_contents_with_popularity.sort_values(by='popularity_score', ascending=False)

    recommended_list = []
    current_count = 0
    for _, row in sorted_popular_contents.iterrows():
        if current_count >= num_recommendations:
            break

        # _create_recommended_content 함수 호출 변경: 딕셔너리 데이터 전달
        recommended_list.append(_create_recommended_content(
            content_data=row.to_dict(), # DataFrame row를 딕셔너리로 변환하여 전달
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

    # 1. 해당 페르소나와 연결된 장르 ID 가져오기
    # 'match_score' 대신 persona_genres_df에 해당 persona_id가 존재하는지 확인
    persona_matched_genres_df = pbr_app_state.persona_genres_df[
        (pbr_app_state.persona_genres_df['persona_id'] == persona_id)
    ]

    if persona_matched_genres_df.empty:
        logger.info(f"페르소나 ID {persona_id}에 매칭되는 장르가 없습니다. 전체 인기 콘텐츠로 폴백합니다.")
        return _get_general_popular_recommendations(user_id=None, num_recommendations=num_recommendations) # user_id는 여기서는 의미 없으므로 None 전달

    persona_genre_ids = persona_matched_genres_df['genre_id'].unique().tolist()
    
    if not persona_genre_ids:
        logger.info(f"페르소나 ID {persona_id}에 연결된 장르 ID가 없습니다. 전체 인기 콘텐츠로 폴백합니다.")
        return _get_general_popular_recommendations(user_id=None, num_recommendations=num_recommendations)

    # 2. 페르소나 장르에 해당하는 콘텐츠 필터링
    # contents_df의 'genres' 컬럼이 리스트 형태로 되어 있다고 가정
    # 각 콘텐츠의 장르 리스트와 페르소나 장르 리스트의 교집합이 하나라도 있으면 매칭
    
    # 먼저 genreId_orig를 통해 genre name을 얻기 위한 매핑 생성
    genre_id_to_name_map = pbr_app_state.genres_df.set_index('id')['name'].to_dict()
    persona_genre_names = [genre_id_to_name_map[gid] for gid in persona_genre_ids if gid in genre_id_to_name_map]

    if not persona_genre_names:
        logger.info(f"페르소나 ID {persona_id}에 연결된 유효한 장르 이름이 없습니다. 전체 인기 콘텐츠로 폴백합니다.")
        return _get_general_popular_recommendations(user_id=None, num_recommendations=num_recommendations)

    # DataFrame apply를 사용하여 각 콘텐츠의 'genres' 리스트와 페르소나 장르를 비교
    # `genres` 컬럼이 문자열로 저장되어 있다면 ast.literal_eval 등으로 리스트로 변환 필요
    # 현재 `generate_unified_user_feedback`에서 'genres'를 리스트로 변환하고 있으므로,
    # contents_df에도 동일한 로직이 적용되어야 합니다.
    
    # Assuming contents_df['genres'] is already a list or can be safely converted
    # If not, add: contents_df['genres'] = contents_df['genres'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    # 로드 시점에 이미 처리되었다고 가정하고 진행 (contents_df 로딩 부분에서 변환하는 것이 이상적)
    
    matched_contents = pbr_app_state.contents_df[
        pbr_app_state.contents_df['genres'].apply(
        lambda content_genres: bool(set(content_genres if content_genres is not None else []).intersection(persona_genre_names))
        )   
    ]

    # 1. 페르소나의 'excluded_genres' 가져오기 (이 데이터가 persona_details_map에 미리 로드되어 있어야 합니다.)
    persona_excluded_genre_names: List[str] = []
    persona_detail = pbr_app_state.persona_details_map.get(persona_id)
    if persona_detail and 'excluded_genres' in persona_detail:
        persona_excluded_genre_names = persona_detail['excluded_genres']
        logger.debug(f"페르소나 ID {persona_id}의 제외 장르: {persona_excluded_genre_names}")

    # 2. 제외 장르 필터링
    if persona_excluded_genre_names and not matched_contents.empty:
        initial_matched_count = matched_contents.shape[0]
        
        # 콘텐츠의 장르가 제외 장르 중 하나라도 포함하는 경우 필터링
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

    # 3. 필터링된 콘텐츠의 인기 점수 계산 및 정렬
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
    # .head(num_recommendations)를 여기서 바로 적용하면, 필터링 후 개수가 부족할 수 있으므로,
    # 아래 루프에서 제어하는 것이 더 유연합니다.

    recommended_list = []
    current_count = 0 # 현재 추가된 추천 개수를 세는 변수
    for _, row in top_popular_contents.iterrows():
        if current_count >= num_recommendations:
            break

        # content_id = int(row['content_id']) # 이 줄은 이제 필요 없습니다.
        # content_type = row['type'] # 이 줄도 이제 필요 없습니다.
        
        # _create_recommended_content 함수 호출 변경: 딕셔너리 데이터 전달
        content_info = _create_recommended_content(
            content_data=row.to_dict(), # DataFrame row를 딕셔너리로 변환하여 전달
            predicted_rating=row['popularity_score'],
            persona_genre_match=True
        )
        
        if content_info.contentId is not None:
            recommended_list.append(content_info)
            current_count += 1

    # 만약 원하는 개수를 채우지 못했다면, 폴백 로직이 실행될 수 있습니다.
    # 현재 코드에서는 return _get_general_popular_recommendations(user_id=None, num_recommendations=num_recommendations)
    # 으로 처리하고 있습니다.
    
    logger.info(f"페르소나 ID {persona_id}를 위한 인기 기반 추천 {len(recommended_list)}개 생성 완료.")
    return recommended_list


def update_and_get_recommendations(
    user_id: int,
    db: Session,
    num_recommendations: int = RECOMMENDATION_COUNT,
    content_type_filter: Optional[str] = None) -> List[RecommendedContent]:
    """
    주어진 사용자 ID에 대한 추천 목록을 업데이트하고 반환합니다.
    캐시된 페르소나 기반 추천을 우선적으로 사용하며, 필요한 경우 캐시를 업데이트합니다.
    """
    logger.info(f"사용자 {user_id}에 대한 추천 업데이트 및 조회 시작.")

    unified_feedback = generate_unified_user_feedback(user_id, db)
    main_persona_id, sub_persona_id, main_persona_name, sub_persona_name, all_personas_scores = \
        get_hybrid_persona(user_id, unified_feedback, db)

    # If no main persona is determined (e.g., Baby Persona), provide general popular recommendations
    if main_persona_id is None:
        logger.info(f"사용자 {user_id}의 메인 페르소나가 결정되지 않아 일반 인기 추천을 제공합니다.")
        return _get_general_popular_recommendations(user_id, num_recommendations)

    # _get_popular_contents_for_persona는 이제 RecommendedContent 객체 리스트를 반환합니다.
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

    # 캐시된 contentId 목록을 RecommendedContent 객체로 변환하여 반환
    return unique_recommendations[:num_recommendations]

def get_persona_based_popular_fallback_recommendations(
    user_id: int,
    main_persona_id: int,
    num_recommendations: int, # 이 값은 'persona_router'에서 계산된 필요한 나머지 개수입니다.
    db: Session,
    content_type_filter: Optional[str] = None
) -> List[RecommendedContent]:
    """
    페르소나 기반 인기 추천을 제공합니다.
    주어진 main_persona_id에 해당하는 장르를 기반으로 인기 콘텐츠를 필터링합니다.
    _get_popular_contents_for_persona 함수를 활용합니다.
    """
    logger.info(f"사용자 {user_id}를 위한 페르소나 {main_persona_id} 기반 인기 추천 생성 중 (요청 개수: {num_recommendations}, 유형 필터: {content_type_filter}).")

    # 초기 인기 콘텐츠를 충분히 많이 가져와서 필터링 후에도 목표 개수를 채울 수 있도록 합니다.
    # num_recommendations는 이미 필요한 나머지 개수이므로,
    # 여기에 추가적인 여유분을 더해서 가져와야 필터링 후에도 개수가 부족하지 않습니다.
    # 예를 들어, 요청 개수의 5배 또는 최소 50개 정도 (필요에 따라 조절)
    initial_fetch_count = max(num_recommendations * 5, 50) # 최소 50개는 가져오도록
    raw_persona_popular_contents = _get_popular_contents_for_persona(main_persona_id, initial_fetch_count)
    logger.info(f"페르소나 ID {main_persona_id}를 위한 인기 기반 추천 초기 {len(raw_persona_popular_contents)}개 생성 완료 (요청: {initial_fetch_count}).")


    current_filtered_list = []

    # 1. 콘텐츠 타입 필터링
    if content_type_filter:
        for content in raw_persona_popular_contents:
            if content.type == content_type_filter:
                current_filtered_list.append(content)
        logger.info(f"콘텐츠 타입 '{content_type_filter}' 필터링 후 {len(current_filtered_list)}개 남음.")
    else:
        # 타입 필터가 없으면 모든 원본 콘텐츠를 다음 단계로 전달
        current_filtered_list = list(raw_persona_popular_contents)

    # 2. 이미 본 콘텐츠 제외 (User-Item Matrix 기준)
    user_idx = pbr_app_state.user_id_to_idx_map.get(user_id)
    if user_idx is not None and pbr_app_state.user_item_matrix is not None:
        # pbr_app_state.user_item_matrix는 이미 ReactionType.LIKE, ReactionType.DISLIKE, Review.score 등을 통합한 것으로 간주합니다.
        # 즉, 0이 아닌 값들은 사용자가 상호작용한 콘텐츠를 의미합니다.
        viewed_content_indices = pbr_app_state.user_item_matrix[user_idx].nonzero()[1]
        
        # pbr_app_state.idx_to_content_id_map에 content_id가 있는지 확인 필요
        # idx_to_content_id_map이 딕셔너리가 아닌 리스트나 다른 형태일 경우 변경 필요
        viewed_content_ids = {pbr_app_state.idx_to_content_id_map[idx] for idx in viewed_content_indices 
                              if idx in pbr_app_state.idx_to_content_id_map} # 유효한 인덱스만 필터링

        after_viewed_filter = []
        for rec in current_filtered_list:
            # RecommendedContent 스키마에 content_id 필드가 있을 것이라고 가정
            if rec.contentId not in viewed_content_ids:
                after_viewed_filter.append(rec)
        current_filtered_list = after_viewed_filter
        logger.info(f"이미 본 콘텐츠 제외 후 {len(current_filtered_list)}개 남음.")
    else:
        logger.info("사용자-아이템 매트릭스 정보를 찾을 수 없거나 사용자 인덱스가 없습니다. 시청 기록 필터링 건너_get_popular_contents_for_persona.")


    # 3. 마지막으로 요청된 개수(num_recommendations)만큼만 슬라이싱하여 반환
    # 필터링 후 남은 콘텐츠가 요청된 개수보다 적을 수 있으므로, 실제 남은 개수만큼만 반환
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
    (pbr_app_state.user_similarity_df is None or pbr_app_state.user_similarity_df.shape[0] == 0): # 여기가 수정되었습니다.
        logger.info("user_similarity_df가 초기화되지 않아 새로 계산합니다.")
        # calculate_user_similarity는 user_similarity_matrix (numpy array)를 반환합니다.
        # 이를 csr_matrix로 변환하여 저장합니다.
        # calculate_user_similarity 함수의 두 번째 반환값이 csr_matrix 입니다.
        _, user_sim_matrix_sparse = calculate_user_similarity(
            pbr_app_state.user_item_matrix,
            pbr_app_state.user_id_to_idx_map # 이 인자도 필요합니다.
        )
        pbr_app_state.user_similarity_df = user_sim_matrix_sparse # CSR matrix를 저장하도록 변경

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

     # 1. 대상 사용자와 유사한 사용자 찾기
    # user_similarity_df는 이제 CSR matrix 입니다.
    # 해당 사용자의 행을 가져와서 유사도를 추출합니다.
    user_similarities_row = pbr_app_state.user_similarity_df.getrow(user_idx)

    similar_users_info = []
    # CSR matrix에서 0이 아닌 요소만 순회
    for other_user_idx, similarity_score in zip(user_similarities_row.indices, user_similarities_row.data):
        if other_user_idx == user_idx: # 자기 자신 제외
            continue
        if similarity_score > CF_SIMILARITY_THRESHOLD: # 설정된 임계값 이상인 유사 사용자만 고려
            similar_users_info.append((other_user_idx, similarity_score))

    # 유사도가 높은 순으로 정렬
    similar_users_info.sort(key=lambda x: x[1], reverse=True)

    # 2. 대상 사용자가 아직 평가하지 않은 콘텐츠에 대한 예상 점수 계산
    # 예측 점수를 저장할 딕셔너리의 키를 (content_id, type) 튜플로 변경
    predicted_ratings: Dict[Tuple[int, str], float] = defaultdict(float) # <-- 변경
    similarity_sums: Dict[Tuple[int, str], float] = defaultdict(float)   # <-- 변경

    # 사용자가 이미 평가한 콘텐츠 목록
    # user_rated_content_ids는 이제 (content_id, type) 튜플의 집합으로 변경됩니다.
    user_rated_content_ids: Set[Tuple[int, str]] = set() # <-- 변경
    for c_idx in pbr_app_state.user_item_matrix[user_idx].nonzero()[1]:
        original_content_id, content_type_from_matrix = pbr_app_state.idx_to_content_id_map[c_idx]
        user_rated_content_ids.add((original_content_id, content_type_from_matrix)) # <-- 변경


    for other_user_idx, similarity in similar_users_info:
        # 유사한 사용자의 평가 기록 가져오기
        other_user_ratings_indices = pbr_app_state.user_item_matrix[other_user_idx].nonzero()[1]
        for content_matrix_idx in other_user_ratings_indices:
            original_content_id, content_type_from_matrix = pbr_app_state.idx_to_content_id_map[content_matrix_idx]

            # 사용자가 이미 평가한 콘텐츠는 제외 (이제 (content_id, type) 튜플로 비교)
            if (original_content_id, content_type_from_matrix) in user_rated_content_ids: # <-- 변경
                continue

            # 유형 필터 적용
            if content_type_filter and content_type_filter != content_type_from_matrix:
                continue

            rating = pbr_app_state.user_item_matrix[other_user_idx, content_matrix_idx]

            # predicted_ratings와 similarity_sums의 키를 (content_id, type) 튜플로 사용
            predicted_ratings[(original_content_id, content_type_from_matrix)] += rating * similarity # <-- 변경
            similarity_sums[(original_content_id, content_type_from_matrix)] += abs(similarity) # <-- 변경 (음수 유사도도 고려하여 절댓값 사용)

    # 예상 점수 정규화
    final_predicted_ratings: Dict[Tuple[int, str], float] = {} # <-- 변경
    for (content_id, content_type), total_score in predicted_ratings.items(): # <-- 변경
        if similarity_sums[(content_id, content_type)] > 0: # <-- 변경
            final_predicted_ratings[(content_id, content_type)] = total_score / similarity_sums[(content_id, content_type)] # <-- 변경
        else:
            final_predicted_ratings[(content_id, content_type)] = 0 # <-- 변경 (유사 사용자가 없거나 유사도 합이 0인 경우)

    # 3. 예상 점수가 높은 순으로 정렬하여 추천 콘텐츠 목록 생성
    # 정렬 키도 이제 (content_id, type) 튜플을 반환하도록 합니다.
    recommended_content_items = sorted(final_predicted_ratings.items(), key=lambda item: item[1], reverse=True) # <-- 변경된 변수명

    recommendations: List[RecommendedContent] = []
    # seen_content_ids는 이제 (content_id, type) 튜플의 집합으로 변경됩니다.
    seen_content_items: Set[Tuple[int, str]] = set() # <-- 변경

    # recommended_content_items에서 (content_id, content_type)과 predicted_rating을 함께 추출
    for (content_id, content_type), predicted_rating in recommended_content_items: # <-- 변경
        if (content_id, content_type) not in seen_content_items: # <-- 변경
            # content_id와 content_type에 해당하는 정확한 콘텐츠 정보를 pbr_app_state.contents_df에서 가져옵니다.
            content_row_df = pbr_app_state.contents_df[
                (pbr_app_state.contents_df['content_id'] == content_id) &
                (pbr_app_state.contents_df['type'] == content_type) # <--- type 필터 추가
            ]

            if content_row_df.empty:
                logger.warning(f"CF 추천: content_id {content_id}, type {content_type}의 정보를 pbr_app_state.contents_df에서 찾을 수 없습니다. 건너뜜.")
                continue # 정보가 없는 콘텐츠는 건너뛰기

            content_row = content_row_df.iloc[0].to_dict() # DataFrame row를 딕셔너리로 변환

            # _create_recommended_content 함수 호출 변경: 딕셔너리 데이터 전달
            content_info = _create_recommended_content(
                content_data=content_row,
                predicted_rating=predicted_rating
            )
            
            if content_info.contentId is not None:
                recommendations.append(content_info)
                seen_content_items.add((content_id, content_type)) # <-- 변경
        
        if len(recommendations) >= num_recommendations: # 'RECOMMENDATION_COUNT' 대신 'num_recommendations' 사용
            break

    logger.info(f"사용자 {user_id}에게 {len(recommendations)}개의 협업 필터링 추천 생성 완료.")
    return recommendations