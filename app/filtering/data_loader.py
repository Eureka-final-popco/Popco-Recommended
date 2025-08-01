import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import text
from database import SessionLocal 
from .state import fr_app_state 

logger = logging.getLogger(__name__)

def load_all_filtering_data():
    logger.info("필터링 시스템 데이터 로드 시작...")
    
    db: Session = SessionLocal() 
    try:
        # 1. Contents 데이터 로드 (release_date에서 release_year 추출, platform 컬럼 제거)
        contents_query = """
       SELECT
            c.id AS content_id,
            c.title,
            c.type,
            c.rating_average,
            c.poster_path,
            YEAR(c.release_date) AS release_year,
            GROUP_CONCAT(DISTINCT g.name ORDER BY g.name SEPARATOR '|') AS genres,
            GROUP_CONCAT(DISTINCT p.name ORDER BY p.name SEPARATOR '|') AS platforms -- ===> 플랫폼 정보 추가
        FROM contents c
        LEFT JOIN content_genres cg ON c.id = cg.content_id AND c.type = cg.content_type
        LEFT JOIN genres g ON cg.genre_id = g.id 
        LEFT JOIN watch_providers wp ON c.id = wp.content_id AND c.type = wp.type -- ===> watch_providers 조인
        LEFT JOIN providers p ON wp.provider_id = p.id -- ===> providers 조인
        GROUP BY c.id, c.type
        ORDER BY c.id, c.type
        """
        result = db.execute(text(contents_query))
        contents_df = pd.DataFrame(result.fetchall(), columns=result.keys())

        # 'genres' 컬럼 처리
        if 'genres' in contents_df.columns:
            contents_df['genres'] = contents_df['genres'].apply(
                lambda x: [g.strip() for g in x.split('|')] if isinstance(x, str) and x else []
            )
        
        # 'platforms' 컬럼 처리
        if 'platforms' in contents_df.columns:
            contents_df['platforms'] = contents_df['platforms'].apply(
                lambda x: [p.strip() for p in x.split('|')] if isinstance(x, str) and x else []
            )
        else:
            contents_df['platforms'] = [[]] * len(contents_df) # 컬럼이 없을 경우 빈 리스트로 초기화

        fr_app_state.contents_df = contents_df
        logger.info(f"필터링용 Contents 데이터 로드 완료: {len(contents_df)}개 (장르, 연도, 플랫폼 포함).")

        # 2. Genres 데이터 로드 (기존과 동일)
        genres_query = """
        SELECT id AS genre_id, name AS genre_name FROM genres
        """
        result = db.execute(text(genres_query))
        genres_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        fr_app_state.genres_df = genres_df
        fr_app_state.genre_id_to_name_map = pd.Series(genres_df.genre_name.values, index=genres_df.genre_id).to_dict()
        logger.info(f"필터링용 Genres 데이터 로드 및 매핑 완료: {len(genres_df)}개.")

        # 3. Reactions 및 Reviews 데이터 (기존과 동일, 연령대별 인기 계산에 활용)
        result = db.execute(text("SELECT user_id, content_id, reaction, type FROM content_reactions"))
        reactions_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        fr_app_state.reactions_df = reactions_df
        logger.info(f"필터링용 Reactions 데이터 로드 완료: {len(reactions_df)}개.")

        result = db.execute(text("SELECT user_id, content_id, type, score FROM reviews"))
        reviews_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        fr_app_state.reviews_df = reviews_df
        logger.info(f"필터링용 Reviews 데이터 로드 완료: {len(reviews_df)}개.")
        
        # 4. 사용자 데이터 로드 (연령대별 인기 계산을 위해)
        # users 테이블과 user_details 테이블을 조인하여 birthdate를 가져옵니다.
        users_query = """
        SELECT
            u.user_id,
            DATE_FORMAT(ud.birthdate, '%Y-%m-%d') AS birth_date
        FROM
            users u
        JOIN
            user_details ud ON u.user_id = ud.user_id
        """
        result = db.execute(text(users_query))
        users_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        
        # birth_date가 없는 사용자가 있을 수 있으므로 오류 처리 추가 
        users_df['birth_date'] = pd.to_datetime(users_df['birth_date'], errors='coerce') # 유효하지 않은 날짜는 NaT로 변환
        
        # NaT 값이 있는 경우 age 계산에서 제외하거나 기본값 설정
        current_year = pd.Timestamp.now().year
        users_df['age'] = users_df['birth_date'].apply(
            lambda x: current_year - x.year if pd.notna(x) else None
        )
        
        fr_app_state.users_df = users_df[['user_id', 'age']].dropna(subset=['age']) # age가 None인 행 제거
        logger.info(f"필터링용 사용자 데이터 로드 완료: {len(fr_app_state.users_df)}개 (나이 정보 포함).")

        # 5. 페르소나 관련 데이터 로드 (페르소나별 인기 계산을 위해)
        # 5-1. 페르소나 정의 데이터 로드
        personas_query = "SELECT persona_id, name AS persona_name, description FROM personas"
        result = db.execute(text(personas_query))
        persona_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        fr_app_state.persona_df = persona_df
        
        # 페르소나 ID-이름 매핑 생성
        fr_app_state.persona_id_to_name_map = pd.Series(persona_df.persona_name.values, index=persona_df.persona_id).to_dict()
        fr_app_state.persona_name_to_id_map = {v: k for k, v in fr_app_state.persona_id_to_name_map.items()}
        logger.info(f"페르소나 정의 데이터 로드 완료: {len(persona_df)}개.")

        # 5-2. 사용자 페르소나 점수 데이터 로드
        user_personas_query = "SELECT user_id, persona_id, score FROM user_personas"
        result = db.execute(text(user_personas_query))
        fr_app_state.all_user_personas_df = pd.DataFrame(result.fetchall(), columns=result.keys())
        logger.info(f"사용자 페르소나 점수 데이터 로드 완료: {len(fr_app_state.all_user_personas_df)}개.")

        # 6. 매핑 생성 (기존과 동일, 필요하다면)
        all_content_tuples = fr_app_state.contents_df[['content_id', 'type']].drop_duplicates().apply(tuple, axis=1).tolist()
        fr_app_state.content_id_to_idx_map = {content_tuple: idx for idx, content_tuple in enumerate(all_content_tuples)}
        fr_app_state.idx_to_content_id_map = {idx: content_tuple for content_tuple, idx in fr_app_state.content_id_to_idx_map.items()}
        logger.info(f"필터링용 콘텐츠 매핑 완료: {len(all_content_tuples)}개.")

        logger.info("필터링 시스템 모든 데이터 로드 및 fr_app_state 업데이트 완료.")

    except Exception as e:
        logger.error(f"필터링 시스템 데이터 로드 중 오류 발생: {e}", exc_info=True)
        raise
    finally:
        db.close()