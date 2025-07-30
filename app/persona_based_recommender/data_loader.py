# data_loader.py

import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from .state import pbr_app_state
from database import SessionLocal
from .config import LIKE_WEIGHT, DISLIKE_WEIGHT, STAR_RATING_WEIGHTS

logger = logging.getLogger(__name__)

def load_all_data():
    logger.info("모든 데이터 로드 시작...")
    
    db: Session = SessionLocal() 
    try:
        # Contents 데이터 로드 (genres 컬럼 포함하도록 수정)
        contents_query = """
        SELECT
            c.id AS content_id, -- <<< 여기서 contentId_orig를 content_id로 변경!
            c.title,
            c.type,
            c.rating_average,
            c.rating_count,
            c.poster_path,
            GROUP_CONCAT(g.name ORDER BY g.name SEPARATOR '|') AS genres
        FROM contents c
        LEFT JOIN content_genres cg ON c.id = cg.content_id AND c.type = cg.content_type
        LEFT JOIN genres g ON cg.genre_id = g.id 
        GROUP BY c.id, c.type
        ORDER BY c.id, c.type
        """
        result = db.execute(text(contents_query))
        contents_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        if 'genres' in contents_df.columns:
            contents_df['genres'] = contents_df['genres'].apply(
                lambda x: [g.strip() for g in x.split('|')] if isinstance(x, str) and x else []
            )
        pbr_app_state.contents_df = contents_df
        logger.info(f"Contents 데이터 로드 완료: {len(contents_df)}개.")

        # Reactions 데이터 로드 (이전 수정사항 그대로 유지, 잘 작동하는 부분)
        result = db.execute(text("SELECT user_id, content_id, reaction, type FROM content_reactions"))
        reactions_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.reactions_df = reactions_df
        logger.info(f"Reactions 데이터 로드 완료: {len(reactions_df)}개.")

        # Reviews 데이터 로드 (이전 수정사항 그대로 유지, 잘 작동하는 부분)
        result = db.execute(text("SELECT user_id, content_id, type, score FROM reviews"))
        reviews_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.reviews_df = reviews_df
        logger.info(f"Reviews 데이터 로드 완료: {len(reviews_df)}개.")

        # UserPersonas 데이터 로드 (변경 없음)
        result = db.execute(text("SELECT * FROM user_personas"))
        all_user_personas_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.all_user_personas_df = all_user_personas_df
        logger.info(f"UserPersonas 데이터 로드 완료: {len(all_user_personas_df)}개.")

        # Personas 데이터 로드 (변경 없음)
        result = db.execute(text("SELECT * FROM personas"))
        persona_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.persona_df = persona_df
        logger.info(f"Personas 데이터 로드 완료: {len(persona_df)}개.")

        # PersonaGenres 데이터 로드 (변경 없음)
        result = db.execute(text("SELECT * FROM persona_genres"))
        persona_genres_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.persona_genres_df = persona_genres_df
        logger.info(f"personaGenres 데이터 로드 완료: {len(persona_genres_df)}개.")

        # Genres 데이터 로드 (변경 없음)
        result = db.execute(text("SELECT * FROM genres"))
        genres_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.genres_df = genres_df
        
        pbr_app_state.genre_id_to_name_map = pd.Series(genres_df.name.values, index=genres_df.id).to_dict()
        pbr_app_state.genre_name_to_id_map = {v: k for k, v in pbr_app_state.genre_id_to_name_map.items()}
        logger.info(f"Genres 데이터 로드 완료: {len(genres_df)}개.")

        # PersonaOptions 데이터 로드 (변경 없음)
        result = db.execute(text("SELECT * FROM persona_options"))
        persona_options_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.persona_options_df = persona_options_df # 추가

        # Options 데이터 로드 (변경 없음)
        result = db.execute(text("SELECT * FROM options"))
        options_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        pbr_app_state.options_df = options_df # 추가
        
        logger.info(f"PersonaOptions 데이터 로드 완료: {len(persona_options_df)}개.")
        logger.info(f"Options 데이터 로드 완료: {len(options_df)}개.")

        pbr_app_state.user_qa_answers_df = pd.DataFrame(columns=['user_id', 'question_id', 'option_id', 'answer'])
        logger.info("User QA Answers 데이터 로드 (빈 DataFrame으로 초기화) 완료.")

        # ====================================================================
        # 핵심 추가: 사용자 ID 및 콘텐츠 ID 매핑 생성 및 pbr_app_state에 저장
        # ====================================================================
        # 모든 고유 사용자 ID 수집
        all_user_ids = pd.concat([pbr_app_state.reactions_df['user_id'], pbr_app_state.reviews_df['user_id']]).unique()
        pbr_app_state.user_id_to_idx_map = {user_id: idx for idx, user_id in enumerate(all_user_ids)}
        pbr_app_state.user_idx_to_id_map = {idx: user_id for user_id, idx in pbr_app_state.user_id_to_idx_map.items()}
        pbr_app_state.all_user_ids = all_user_ids.tolist()
        logger.info(f"사용자 매핑 완료: {len(all_user_ids)}명.")

        # 모든 고유 콘텐츠 ID (original_id, type) 튜플 수집
        # pbr_app_state.contents_df에서 contentId_orig를 content_id로 변경!
        all_content_tuples = pbr_app_state.contents_df[['content_id', 'type']].drop_duplicates().apply(tuple, axis=1).tolist() # <<< 여기도 content_id로 변경!
        pbr_app_state.content_id_to_idx_map = {content_tuple: idx for idx, content_tuple in enumerate(all_content_tuples)}
        pbr_app_state.idx_to_content_id_map = {idx: content_tuple for content_tuple, idx in pbr_app_state.content_id_to_idx_map.items()}
        logger.info(f"콘텐츠 매핑 완료: {len(all_content_tuples)}개.")


        try:
            # 3. Reactions 데이터 처리 (변경 없음)
            reactions_df_processed = pbr_app_state.reactions_df.copy()

            reactions_df_processed['score'] = reactions_df_processed['reaction'].apply(
                lambda x: LIKE_WEIGHT if x == 'LIKE' else (DISLIKE_WEIGHT if x == 'DISLIKE' else 0)
            )
            reactions_df_processed.rename(columns={'type': 'content_type'}, inplace=True)
            reactions_df_processed = reactions_df_processed[['user_id', 'content_id', 'content_type', 'score']]

            # 2. Reviews 데이터 처리 (변경 없음)
            reviews_df_processed = pbr_app_state.reviews_df.copy()
            reviews_df_processed['score'] = reviews_df_processed['score'].apply(
                lambda x: STAR_RATING_WEIGHTS.get(x, 0.0)
            )
            reviews_df_processed.rename(columns={'type': 'content_type'}, inplace=True)
            reviews_df_processed = reviews_df_processed[['user_id', 'content_id', 'content_type', 'score']]

            # 3. Reactions와 Reviews 데이터 통합 (변경 없음)
            all_interactions_df = pd.concat([reactions_df_processed, reviews_df_processed], ignore_index=True)

            # 4. 사용자 및 콘텐츠 인덱스 매핑 적용 (변경 없음)
            all_interactions_df['user_idx'] = all_interactions_df['user_id'].map(pbr_app_state.user_id_to_idx_map)
            
            all_interactions_df['content_tuple'] = list(zip(all_interactions_df['content_id'], all_interactions_df['content_type']))
            all_interactions_df['content_idx'] = all_interactions_df['content_tuple'].map(pbr_app_state.content_id_to_idx_map)

            all_interactions_df.dropna(subset=['user_idx', 'content_idx', 'score'], inplace=True)
            all_interactions_df['user_idx'] = all_interactions_df['user_idx'].astype(int)
            all_interactions_df['content_idx'] = all_interactions_df['content_idx'].astype(int)

            if not all_interactions_df.empty:
                from scipy.sparse import csr_matrix
                
                user_count = len(pbr_app_state.user_id_to_idx_map)
                content_count = len(pbr_app_state.content_id_to_idx_map)

                pbr_app_state.user_item_matrix = csr_matrix(
                    (
                        all_interactions_df['score'],
                        (all_interactions_df['user_idx'], all_interactions_df['content_idx'])
                    ),
                    shape=(user_count, content_count)
                )
                logger.info(f"CF 사용자-아이템 매트릭스 생성 완료. 형태: {pbr_app_state.user_item_matrix.shape}, 비0 요소: {pbr_app_state.user_item_matrix.nnz}개.")
            else:
                pbr_app_state.user_item_matrix = None
                logger.warning("사용자 상호작용 데이터(LIKE/DISLIKE/별점)가 없어 CF 사용자-아이템 매트릭스를 생성할 수 없습니다.")
        except Exception as e:
            pbr_app_state.user_item_matrix = None
            logger.error(f"CF 사용자-아이템 매트릭스 생성 중 오류 발생: {e}", exc_info=True)

        logger.info(f"CF 매트릭스 확인: user_item_matrix is None? {pbr_app_state.user_item_matrix is None}")
        if pbr_app_state.user_item_matrix is not None:
            logger.info(f"CF 매트릭스 형태: {pbr_app_state.user_item_matrix.shape}")
            logger.info(f"CF 매트릭스 비어있는가?: {pbr_app_state.user_item_matrix.nnz == 0}")

        logger.info(f"사용자 매핑 크기: {len(pbr_app_state.user_id_to_idx_map)}")
        logger.info(f"콘텐츠 매핑 크기: {len(pbr_app_state.content_id_to_idx_map)}")

        test_user_id = 44
        if test_user_id in pbr_app_state.user_id_to_idx_map:
            logger.info(f"사용자 ID {test_user_id} 매핑 인덱스: {pbr_app_state.user_id_to_idx_map[test_user_id]}")
        else:
            logger.info(f"사용자 ID {test_user_id} 매핑 존재하지 않음.")
        # ====================================================================


        # 페르소나 상세 정보 맵 생성 (변경 없음)
        pbr_app_state.persona_id_to_name_map = pd.Series(persona_df.name.values, index=persona_df.persona_id).to_dict()
        pbr_app_state.persona_name_to_id_map = {v: k for k, v in pbr_app_state.persona_id_to_name_map.items()}

        if pbr_app_state.persona_details_map is None:
            pbr_app_state.persona_details_map = {}

        for _, row in persona_df.iterrows():
            persona_id = row['persona_id']
            persona_name = row['name']
            description = row['description']

            keywords = []
            persona_relevant_genres = persona_genres_df[persona_genres_df['persona_id'] == persona_id]
            for _, genre_row in persona_relevant_genres.iterrows():
                genre_id = genre_row['genre_id']
                genre_name = pbr_app_state.genre_id_to_name_map.get(genre_id) 
                if genre_name:
                    keywords.append(genre_name)

            qa_mapping = {}
            persona_qa_options = persona_options_df[persona_options_df['persona_id'] == persona_id]
            merged_qa_data = pd.merge(persona_qa_options, options_df, on=['question_id', 'option_id'], how='left')

            if not merged_qa_data.empty:
                for q_id in merged_qa_data['question_id'].unique():
                    q_data = merged_qa_data[merged_qa_data['question_id'] == q_id]
                    if not q_data.empty:
                        best_option_row = q_data.loc[q_data['score'].idxmax()]
                        qa_mapping[str(q_id)] = str(best_option_row['content'])

            pbr_app_state.persona_details_map[persona_id] = {
                'persona_name': persona_name,
                'description': description,
                'keywords': keywords,
                'qa_mapping': qa_mapping
            }

        logging.info("모든 데이터 로드 및 pbr_app_state 업데이트 완료.")

        print(pbr_app_state.persona_details_map.get(4, {}).get('persona_name'))
        print(pbr_app_state.persona_details_map.get(4, {}).get('keywords'))

    except Exception as e:
        logging.error(f"데이터 로드 중 오류 발생: {e}", exc_info=True)
        raise

    finally:
        db.close()