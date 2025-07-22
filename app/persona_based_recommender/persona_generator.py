import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import itertools
from collections import defaultdict
from .config import (
    LIKE_WEIGHT,
    DISLIKE_WEIGHT,
    STAR_RATING_WEIGHTS,
    POSITIVE_FEEDBACK_THRESHOLD,
    QA_WEIGHT,
    CONTENT_FEEDBACK_WEIGHT,
    MIN_FEEDBACK_FOR_PERSONA_DETERMINATION,
    MIN_POSITIVE_RATINGS_FOR_GROWTH,
    CF_SIMILARITY_THRESHOLD,
    PENALTY_FOR_NO_PERSONA_MATCH,
    default_personas,
    content_genre_keywords_mapping
)


GLOBAL_USER_ID_TO_IDX: Dict[int, int] = {}
GLOBAL_USER_IDX_TO_ID: Dict[int, int] = {}
GLOBAL_CONTENT_ID_TO_IDX: Dict[int, int] = {}
GLOBAL_CONTENT_IDX_TO_ID: Dict[int, int] = {}


def _get_user_idx_from_id(user_id: int) -> int:
    return user_id

def _get_user_id_from_idx(idx: int) -> int:
    return idx

def _get_content_idx_from_id(content_id: int) -> int:
    return content_id

def _get_content_id_from_idx(idx: int) -> int:
    return idx


# 사용자 통합 피드백 생성 함수
def generate_unified_user_feedback(
    user_id: int,
    reactions_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    contents_df: pd.DataFrame
) -> Dict[int, float]:
    user_reactions = reactions_df[reactions_df['userId'] == user_id]
    user_reviews = reviews_df[reviews_df['userId'] == user_id]

    unified_feedback = {}

    if 'updated_at' in user_reactions.columns:
        user_reactions = user_reactions.sort_values(by='updated_at', ascending=True)
    elif 'created_at' in user_reactions.columns:
        user_reactions = user_reactions.sort_values(by='created_at', ascending=True)
    
    if 'updated_at' in user_reviews.columns:
        user_reviews = user_reviews.sort_values(by='updated_at', ascending=True)
    elif 'created_at' in user_reviews.columns:
        user_reviews = user_reviews.sort_values(by='created_at', ascending=True)

    # 1. 좋아요/싫어요 처리 (가장 최근 피드백만 반영)
    for _, row in user_reactions.iterrows():
        content_id = row['contentId']
        reaction_type = row['reaction']
        
        if reaction_type == '좋아요':
            unified_feedback[content_id] = LIKE_WEIGHT
        elif reaction_type == '싫어요':
            unified_feedback[content_id] = DISLIKE_WEIGHT

    # 2. 리뷰 (별점) 처리
    for _, row in user_reviews.iterrows():
        content_id = row['contentId']
        score = row['score']
        rating_weight = STAR_RATING_WEIGHTS.get(score, 0.0) 
        unified_feedback[content_id] = rating_weight 

    return unified_feedback

# 초기 사용자 페르소나 생성/업데이트 함수
def generate_initial_user_personas(
    user_ids: List[int],
    reactions_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    contents_df: pd.DataFrame,
    default_personas: Dict[str, Dict[str, Any]],
    content_genre_keywords_mapping: Dict[str, List[str]],
    initial_answers: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    all_user_personas_data = []

    for user_id in user_ids:
        # 1. 사용자 통합 피드백 생성
        unified_feedback = generate_unified_user_feedback(user_id, reactions_df, reviews_df, contents_df)

        # 2. 페르소나별 점수 계산
        persona_raw_scores = {persona_id: 0 for persona_id in default_personas.keys()} 
        
        # 긍정적 피드백 수
        positive_feedback_count = sum(1 for score in unified_feedback.values() if score >= POSITIVE_FEEDBACK_THRESHOLD)

        # 초기 답변이 없거나 긍정적 피드백 수가 MIN_POSITIVE_RATINGS_FOR_GROWTH 미달 시 페르소나 추정 안 함
        if not initial_answers and positive_feedback_count < MIN_POSITIVE_RATINGS_FOR_GROWTH:
            for persona_id in default_personas.keys():
                all_user_personas_data.append({
                    'user_id': user_id,
                    'persona_id': persona_id,
                    'score': 0.0 
                })
            continue 

        # 2.1. 초기 설문 답변 기반 점수
        if initial_answers:
            for persona_id, persona_data in default_personas.items():
                for question_id, expected_answer in persona_data.get('qa_mapping', {}).items():
                    if initial_answers.get(question_id) == expected_answer:
                        persona_raw_scores[persona_id] += 1 

        # 2.2. 콘텐츠 피드백 기반 점수
        positive_feedback_contents_ids = [cid for cid, score in unified_feedback.items() if score >= POSITIVE_FEEDBACK_THRESHOLD]

        if positive_feedback_contents_ids: 
            for content_id in positive_feedback_contents_ids:
                content_info_row = contents_df[contents_df['contentId'] == content_id]

                if not content_info_row.empty:
                    content_genres_data = content_info_row['genres'].iloc[0]

                    if isinstance(content_genres_data, str):
                        content_genres = [g.strip() for g in content_genres_data.split('|')]
                    elif isinstance(content_genres_data, list):
                        content_genres = [g.strip() for g in content_genres_data]
                    else:
                        content_genres = []
                else:
                    content_genres = []

                for persona_id, persona_data in default_personas.items():
                    persona_keywords = persona_data.get('keywords', [])

                    persona_actual_genres = set()
                    for keyword in persona_keywords:
                        if keyword in content_genre_keywords_mapping:
                            persona_actual_genres.update(content_genre_keywords_mapping[keyword])
                        else:
                            persona_actual_genres.add(keyword)

                    match_count = len(set(content_genres).intersection(persona_actual_genres))
                    if match_count > 0:
                        persona_raw_scores[persona_id] += match_count 

        # 3. 가중치 적용 및 최종 페르소나 점수 정규화
        final_persona_scores = {}
        
        max_qa_score_possible = sum(len(p.get('qa_mapping', {})) for p in default_personas.values()) if initial_answers else 0
        
        max_content_score_possible = 0
        if positive_feedback_contents_ids:
            for content_id in positive_feedback_contents_ids:
                content_info_row = contents_df[contents_df['contentId'] == content_id]
                if not content_info_row.empty:
                    content_genres_data = content_info_row['genres'].iloc[0]
                    content_genres_set = set()
                    if isinstance(content_genres_data, str):
                        content_genres_set = set([g.strip() for g in content_genres_data.split('|')])
                    elif isinstance(content_genres_data, list):
                        content_genres_set = set([g.strip() for g in content_genres_data])

                    max_match_for_content = 0
                    for persona_data in default_personas.values():
                        persona_keywords = persona_data.get('keywords', [])
                        persona_actual_genres_set = set()
                        for keyword in persona_keywords:
                            if keyword in content_genre_keywords_mapping:
                                persona_actual_genres_set.update(content_genre_keywords_mapping[keyword])
                            else:
                                persona_actual_genres_set.add(keyword)
                        match_count = len(content_genres_set.intersection(persona_actual_genres_set))
                        max_match_for_content = max(max_match_for_content, match_count)
                    max_content_score_possible += max_match_for_content


        for persona_id, raw_score in persona_raw_scores.items():
            qa_normalized_score = 0
            if initial_answers and max_qa_score_possible > 0:
                qa_matches_for_persona = sum(1 for qid, ans in default_personas[persona_id].get('qa_mapping', {}).items() if initial_answers.get(qid) == ans)
                qa_normalized_score = (qa_matches_for_persona / max_qa_score_possible)

            content_normalized_score = 0
            if positive_feedback_contents_ids and max_content_score_possible > 0:
                content_normalized_score = (raw_score / max_content_score_possible) 
                
            final_score_for_persona = (qa_normalized_score * QA_WEIGHT) + (content_normalized_score * CONTENT_FEEDBACK_WEIGHT)
            final_persona_scores[persona_id] = final_score_for_persona

        # 모든 점수가 0인 경우 (피드백이 너무 적거나 매칭이 안되는 경우)
        if not any(final_persona_scores.values()):
            num_personas = len(default_personas)
            final_persona_scores = {p_id: 1.0 / num_personas for p_id in default_personas.keys()} # 균등 배분
        else:
            sum_of_final_scores = sum(final_persona_scores.values())
            if sum_of_final_scores > 0:
                final_persona_scores = {p_id: score / sum_of_final_scores for p_id, score in final_persona_scores.items()}
            else:
                num_personas = len(default_personas)
                final_persona_scores = {p_id: 1.0 / num_personas for p_id in default_personas.keys()}

        # 결과 데이터프레임에 추가
        for persona_id, score in final_persona_scores.items():
            all_user_personas_data.append({
                'user_id': user_id,
                'persona_id': persona_id,
                'score': score
            })

    return pd.DataFrame(all_user_personas_data)


# 하이브리드 페르소나 결정
def get_hybrid_persona(
    user_id: int,
    all_user_personas_df: pd.DataFrame,
    unified_user_feedback: Dict[int, float], 
    contents_df: pd.DataFrame, 
    default_personas: Dict[str, Dict[str, Any]],
    content_genre_keywords_mapping: Dict[str, List[str]]
) -> Tuple[str, str, Dict[str, float]]:
    user_persona_data = all_user_personas_df[all_user_personas_df['user_id'] == user_id]

    if user_persona_data.empty:
        return "알 수 없음", "알 수 없음", {}

    sorted_personas_by_score = sorted(user_persona_data.set_index('persona_id')['score'].to_dict().items(),
                                      key=lambda item: item[1], reverse=True)

    main_persona_id = "알 수 없음"
    sub_persona_id = "알 수 없음"
    all_personas_scores = dict(sorted_personas_by_score)

    if not sorted_personas_by_score or sorted_personas_by_score[0][1] <= 0:
        return "알 수 없음", "알 수 없음", all_personas_scores

    main_score = sorted_personas_by_score[0][1]
    top_score_personas = [item for item in sorted_personas_by_score if item[1] == main_score]

    if len(top_score_personas) > 1:
        highest_rated_content_id = None
        max_feedback_score = -float('inf')

        positive_feedback_contents = {
            cid: score for cid, score in unified_user_feedback.items()
            if score >= POSITIVE_FEEDBACK_THRESHOLD
        }

        if not positive_feedback_contents:
            top_score_personas.sort(key=lambda item: item[0])
            main_persona_id = top_score_personas[0][0]
            if len(top_score_personas) > 1:
                sub_persona_id = top_score_personas[1][0]
            else:
                sub_persona_id = "없음"
            return main_persona_id, sub_persona_id, all_personas_scores

        for content_id, score in positive_feedback_contents.items():
            if score > max_feedback_score:
                max_feedback_score = score
                highest_rated_content_id = content_id
            elif score == max_feedback_score and highest_rated_content_id is not None and content_id < highest_rated_content_id:
                highest_rated_content_id = content_id

        if highest_rated_content_id is not None:
            content_info_row = contents_df[contents_df['contentId'] == highest_rated_content_id]
            if not content_info_row.empty:
                best_content_genres_data = content_info_row['genres'].iloc[0]

                if isinstance(best_content_genres_data, str):
                    best_content_genres = [g.strip() for g in best_content_genres_data.split('|')]
                elif isinstance(best_content_genres_data, list):
                    best_content_genres = [g.strip() for g in best_content_genres_data]
                else:
                    best_content_genres = []
            else:
                best_content_genres = []

            persona_match_counts = {}
            for persona_id, _ in top_score_personas:
                persona_data = default_personas[persona_id]
                persona_keywords = persona_data.get('keywords', [])

                persona_actual_genres_set = set()
                for keyword in persona_keywords:
                    if keyword in content_genre_keywords_mapping:
                        persona_actual_genres_set.update(content_genre_keywords_mapping[keyword])
                    else:
                        persona_actual_genres_set.add(keyword)

                match_count = len(set(best_content_genres).intersection(persona_actual_genres_set))
                persona_match_counts[persona_id] = match_count

            sorted_by_match = sorted(persona_match_counts.items(), key=lambda item: (-item[1], item[0]))
            main_persona_id = sorted_by_match[0][0]

            if len(sorted_by_match) > 1:
                sub_persona_id = sorted_by_match[1][0]
            else:
                sub_persona_id = "없음"

        else: 
            top_score_personas.sort(key=lambda item: item[0])
            main_persona_id = top_score_personas[0][0]
            if len(top_score_personas) > 1:
                sub_persona_id = top_score_personas[1][0]
            else:
                sub_persona_id = "없음"
    else: 
        main_persona_id = top_score_personas[0][0]
        if len(sorted_personas_by_score) > 1:
            sub_persona_id = sorted_personas_by_score[1][0]
        else:
            sub_persona_id = "없음"

    user_total_feedback_count = len(unified_user_feedback)
    if user_total_feedback_count < MIN_FEEDBACK_FOR_PERSONA_DETERMINATION:
        main_persona_name_with_prefix = f"아기_{main_persona_id}" 
    else:
        main_persona_name_with_prefix = main_persona_id

    return main_persona_name_with_prefix, sub_persona_id, all_personas_scores


# 사용자 - 콘텐츠 행렬 생성 (희소 행렬 버전)
def calculate_user_content_matrix_sparse(
    reactions_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    contents_df: pd.DataFrame 
) -> csr_matrix:
    all_user_ids = pd.concat([reactions_df['userId'], reviews_df['userId']]).unique()
    all_content_ids = pd.concat([reactions_df['contentId'], reviews_df['contentId']]).unique()

    if all_user_ids.size == 0 or all_content_ids.size == 0:
        return csr_matrix((0, 0), dtype=np.float64)

    max_user_id = all_user_ids.max()
    max_content_id = all_content_ids.max()

    num_users = int(max_user_id) + 1
    num_contents = int(max_content_id) + 1

    rows = []
    cols = []
    data = []

    unique_users_in_feedback = pd.concat([reactions_df['userId'], reviews_df['userId']]).unique()

    for user_id in unique_users_in_feedback:
        unified_feedback_for_user = generate_unified_user_feedback(user_id, reactions_df, reviews_df, contents_df)
        for content_id, score in unified_feedback_for_user.items():
            rows.append(user_id)
            cols.append(content_id)
            data.append(score)

    user_content_matrix = csr_matrix((data, (rows, cols)), shape=(num_users, num_contents), dtype=np.float64)

    return user_content_matrix

# 사용자 유사도 계산
def calculate_user_similarity(user_content_matrix: csr_matrix) -> pd.DataFrame:
    if user_content_matrix.shape[0] < 2 or user_content_matrix.shape[1] < 1:
        print("경고: 유사도 계산을 위한 충분한 사용자 또는 콘텐츠 데이터가 없습니다.")
        return pd.DataFrame()

    user_similarity = cosine_similarity(user_content_matrix)
    user_similarity_df = pd.DataFrame(user_similarity)

    np.fill_diagonal(user_similarity_df.values, 0)

    user_ids = [_get_user_id_from_idx(i) for i in range(user_content_matrix.shape[0])]
    user_similarity_df.index = user_ids
    user_similarity_df.columns = user_ids

    return user_similarity_df

# 협업 필터링 기반 콘텐츠 추천
def recommend_contents_cf(
    target_user_id: int,
    user_content_matrix: csr_matrix,
    user_similarity_df: pd.DataFrame,
    contents_df_persona: pd.DataFrame,
    all_user_personas_df: pd.DataFrame,
    reactions_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    num_recommendations: int = 10
) -> List[Dict[str, Any]]:
    target_user_idx = target_user_id

    if target_user_idx >= user_content_matrix.shape[0]:
        print(f"경고: 타겟 사용자 ID {target_user_id}에 대한 행렬 범위 정보가 부족합니다. 사용자 ID가 행렬의 최대 인덱스를 초과합니다.")
        return []

    if target_user_id not in user_similarity_df.index:
        print(f"경고: 사용자 {target_user_id}에 대한 유사도 데이터가 없습니다.")
        return []

    # 1. 대상 사용자의 메인 페르소나 식별
    unified_feedback_for_target_user = generate_unified_user_feedback(
        target_user_id, reactions_df, reviews_df, contents_df_persona
    )
    main_persona_name_with_prefix, _, all_persona_scores_dict = get_hybrid_persona(
        target_user_id,
        all_user_personas_df,
        unified_feedback_for_target_user,
        contents_df_persona,
        default_personas,
        content_genre_keywords_mapping
    )
    
    main_persona_pure_id = main_persona_name_with_prefix.replace("아기_", "")

    if main_persona_name_with_prefix == "알 수 없음":
        print(f"사용자 {target_user_id}의 메인 페르소나를 식별할 수 없습니다. 일반 추천을 시도하거나 다른 방식으로 폴백하세요.")
        return []

    print(f"사용자 {target_user_id}'의 메인 페르소나: {main_persona_name_with_prefix} (순수: {main_persona_pure_id})")

    # 2. 해당 메인 페르소나에 속하는 사용자들 필터링
    main_persona_for_each_user = all_user_personas_df.loc[
        all_user_personas_df.groupby('user_id')['score'].idxmax()
    ].copy() 

    relevant_persona_users_df = main_persona_for_each_user[
        main_persona_for_each_user['persona_id'].str.contains(main_persona_pure_id)
    ]
    
    persona_group_user_ids = relevant_persona_users_df['user_id'].tolist()
    
    if target_user_id not in persona_group_user_ids:
        persona_group_user_ids.append(target_user_id)
    
    persona_group_user_ids_in_similarity_df = [
        uid for uid in persona_group_user_ids if uid in user_similarity_df.index
    ]
    
    if len(persona_group_user_ids_in_similarity_df) < 2:
        print(f"경고: {main_persona_name_with_prefix} 페르소나 그룹 내에 유사도 계산을 위한 충분한 사용자(2명 이상)가 없습니다.")
        return []

    # 3. 페르소나 그룹 내에서 타겟 사용자와 유사한 사용자 찾기
    similarities_with_group = user_similarity_df.loc[target_user_id, persona_group_user_ids_in_similarity_df]
    
    if target_user_id in similarities_with_group.index:
        similarities_with_group = similarities_with_group.drop(target_user_id)

    # CF_SIMILARITY_THRESHOLD를 넘는 유사도를 가진 사용자만 선택
    top_similar_users = similarities_with_group[similarities_with_group >= CF_SIMILARITY_THRESHOLD].sort_values(ascending=False)
    
    if top_similar_users.empty:
        print(f"경고: {main_persona_name_with_prefix} 그룹에서 사용자 {target_user_id}와 유사도 임계값({CF_SIMILARITY_THRESHOLD})을 넘는 사용자를 찾을 수 없습니다.")
        return []
            
    selected_similar_user_ids = top_similar_users.index.tolist()

    # 4. 유사 사용자들이 선호하는 콘텐츠 찾기 및 가중 평점 계산
    predicted_ratings = defaultdict(float) 

    watched_content_ids = set(unified_feedback_for_target_user.keys())

    for similar_user_id in selected_similar_user_ids:
        similarity_score = top_similar_users.loc[similar_user_id] 

        sim_user_ratings_row = user_content_matrix[similar_user_id, :]
        rated_content_indices, rated_content_scores = sim_user_ratings_row.nonzero()[1], sim_user_ratings_row.data
        
        for i, content_idx in enumerate(rated_content_indices):
            content_id = _get_content_id_from_idx(content_idx)
            
            if content_id not in watched_content_ids:
                rating_by_similar_user = rated_content_scores[i]
                predicted_ratings[content_id] += similarity_score * rating_by_similar_user
    
    # 5. 정규화 
    sum_of_similarities = top_similar_users.sum()
    if sum_of_similarities == 0:
        predicted_ratings_normalized = {}
    else:
        predicted_ratings_normalized = {
            content_id: score / sum_of_similarities
            for content_id, score in predicted_ratings.items()
        }

    # 6. 페르소나 장르 매칭을 통한 추가 보정 및 최종 추천 목록 생성
    final_recommendations_with_scores = []
    
    # 현재 사용자의 메인 페르소나의 장르 키워드 가져오기
    main_persona_genres_keywords = set()
    if main_persona_pure_id in default_personas:
        main_persona_genres_keywords.update(default_personas[main_persona_pure_id].get('keywords', []))
    
    main_persona_actual_genres = set()
    for keyword in main_persona_genres_keywords:
        if keyword in content_genre_keywords_mapping:
            main_persona_actual_genres.update(content_genre_keywords_mapping[keyword])
        else:
            main_persona_actual_genres.add(keyword)

    sorted_recommendations = sorted(predicted_ratings_normalized.items(), key=lambda item: item[1], reverse=True)

    for content_id, predicted_rating in sorted_recommendations:
        if len(final_recommendations_with_scores) >= num_recommendations:
            break

        content_details = contents_df_persona[contents_df_persona['contentId'] == content_id]
        
        if not content_details.empty:
            content_row = content_details.iloc[0]
            
            content_genres_data = content_row['genres']
            if isinstance(content_genres_data, str):
                content_genres = [g.strip() for g in content_genres_data.split('|')]
            elif isinstance(content_genres_data, list):
                content_genres = [g.strip() for g in content_genres_data]
            else:
                content_genres = []

            persona_genre_match = bool(set(content_genres).intersection(main_persona_actual_genres))
            
            # 페르소나 장르 불일치 시 페널티 적용
            final_score = predicted_rating
            if not persona_genre_match:
                final_score += PENALTY_FOR_NO_PERSONA_MATCH

            final_recommendations_with_scores.append({
                "contentId": int(content_id),
                "title": content_row['title'],
                "genres": content_genres,
                "predicted_rating": float(final_score),
                "persona_genre_match": persona_genre_match
            })
        
    final_recommendations_with_scores.sort(key=lambda x: x['predicted_rating'], reverse=True)
    
    return final_recommendations_with_scores[:num_recommendations]
