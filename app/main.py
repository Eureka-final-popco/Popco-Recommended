import os
import sys
import io
from dotenv import load_dotenv
load_dotenv()
# main.py
import boto3
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime
from typing import List
import pandas as pd
import numpy as np
import json
import traceback
from contextlib import asynccontextmanager
import pandas as pd
import logging
import gc  # 가비지 컬렉션을 위해 추가

from config import settings
from database import check_db_connection, get_db
from popcorithm.models import (
    MovieRecommendation, 
    RecommendationResponse
)

from popcorithm.create_popcorithm_csv import create_movie_metadata_csv
from popcorithm.get_user_pattern import get_user_recent_activities
from popcorithm.popcorithm import calculate_user_preferences
from popcorithm.calc_cosine import calculate_recommendations, load_movie_metadata_with_vectors
from content_based_recommender.contents_recommender_py import ImprovedMovieRecommendationSystem
from content_based_recommender.schemas import RecommendRequest, ContentRecommendationListResponse, RecommendationListResponse
from content_based_recommender.data_saver import get_existing_recommendations, save_recommendations_to_db, get_top_ranked_content, build_recommendation_responses, check_user_exists

# from .popcorithm.create_popcorithm_csv import create_movie_metadata_csv, create_metadata_only_csv
# from .popcorithm.get_user_pattern import get_user_recent_activities
# from .popcorithm.popcorithm import calculate_user_preferences
# from .popcorithm.calc_cosine import calculate_recommendations, load_movie_metadata_with_vectors

from persona_based_recommender.state import pbr_app_state
from persona_based_recommender.persona_router import persona_recommender_router
from persona_based_recommender.data_loader import load_all_data

from filtering.filtering_router import filtering_recommender_router
from filtering.data_loader import load_all_filtering_data

APP_ROOT_DIR = Path(__file__).parent # '/app'
S3_BUCKET_NAME = settings.MY_AWS_S3_BUCKET_NAME
s3 = boto3.client(
    's3',
    aws_access_key_id=settings.MY_AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.MY_AWS_SECRET_ACCESS_KEY,
    region_name=settings.MY_AWS_S3_REGION
)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cached_movie_df = None
cached_features = None
cached_movie_vectors = None

def load_csv_from_s3(bucket_name: str, key: str, chunksize: int = 10000) -> pd.DataFrame:
    """S3에서 CSV 파일을 메모리 효율적으로 읽어서 DataFrame으로 반환"""
    try:
        logger.info(f"S3에서 CSV 로드 시작: s3://{bucket_name}/{key} (청크 크기: {chunksize})")
        
        # S3에서 파일 객체 가져오기
        response = s3.get_object(Bucket=bucket_name, Key=key)
        
        # 바이너리 데이터를 BytesIO로 변환
        csv_data = io.BytesIO(response['Body'].read())
        
        # 청크 단위로 읽어서 메모리 효율적으로 처리
        chunk_list = []
        total_rows = 0
        
        logger.info("청크 단위로 CSV 데이터 읽기 시작...")
        
        # pandas의 chunksize 파라미터 사용
        for chunk_num, chunk in enumerate(pd.read_csv(csv_data, chunksize=chunksize), 1):
            chunk_list.append(chunk)
            total_rows += len(chunk)
            
            if chunk_num % 10 == 0:  # 10청크마다 로그 출력
                logger.info(f"청크 {chunk_num} 처리 완료, 누적 행 수: {total_rows}")
                
            # 메모리 관리: 중간에 가비지 컬렉션
            if chunk_num % 50 == 0:
                gc.collect()
        
        logger.info(f"모든 청크 읽기 완료. DataFrame 합치기 시작...")
        
        # 모든 청크를 하나의 DataFrame으로 합치기
        df = pd.concat(chunk_list, ignore_index=True)
        
        # 메모리 정리
        del chunk_list
        gc.collect()
        
        logger.info(f"S3 CSV 로드 성공: {len(df)}개 행, 메모리 사용량 최적화 완료")
        return df
        
    except Exception as e:
        logger.error(f"S3 CSV 로드 실패: {e}")
        # 메모리 정리
        gc.collect()
        raise HTTPException(status_code=500, detail=f"S3에서 CSV 로드 실패: {e}")

def load_json_from_s3(bucket_name: str, key: str, check_size: bool = True) -> dict:
    """S3에서 JSON 파일을 메모리 효율적으로 읽어서 dict로 반환"""
    try:
        logger.info(f"S3에서 JSON 로드 시작: s3://{bucket_name}/{key}")
        
        # 파일 크기 먼저 확인 (옵션)
        if check_size:
            head_response = s3.head_object(Bucket=bucket_name, Key=key)
            file_size_mb = head_response['ContentLength'] / (1024 * 1024)
            logger.info(f"JSON 파일 크기: {file_size_mb:.2f} MB")
            
            # 큰 파일의 경우 경고 로그
            if file_size_mb > 100:  # 100MB 이상
                logger.warning(f"큰 JSON 파일 감지 ({file_size_mb:.2f} MB). 메모리 사용량을 모니터링하세요.")
        
        # S3에서 파일 객체 가져오기
        response = s3.get_object(Bucket=bucket_name, Key=key)
        
        # JSON 데이터 파싱
        json_content = response['Body'].read().decode('utf-8')
        json_data = json.loads(json_content)
        
        # 메모리 정리
        del json_content
        gc.collect()
        
        logger.info(f"S3 JSON 로드 성공")
        return json_data
        
    except Exception as e:
        logger.error(f"S3 JSON 로드 실패: {e}")
        # 메모리 정리
        gc.collect()
        raise HTTPException(status_code=500, detail=f"S3에서 JSON 로드 실패: {e}")
    
async def initialize_local_recommender_system():
    global recommender_system, movies_dataframe

    logger.info("🔧 로컬 추천 시스템 초기화 시작...")

    try:
        # Docker 환경에서는 /app이 작업 디렉토리
        current_file_dir = Path(__file__).resolve().parent  # /app
        
        # data_processing 디렉토리가 /app과 같은 레벨에 있다면
        data_file_path = current_file_dir / 'content_based_recommender' / 'content_data.csv'  # /app/data_processing/content_data.csv
        
        # 또는 상위 디렉토리에 있다면 이렇게:
        # project_root = current_file_dir.parent  # Docker에서는 필요 없을 수 있음
        # data_file_path = project_root / 'data_processing' / 'content_data.csv'
        
        abs_data_file_path = str(data_file_path)

        logger.info(f"데이터 파일 로드 시도: {abs_data_file_path}")

        movies_dataframe = pd.read_csv(abs_data_file_path, encoding='utf-8-sig', header=0)

        recommender_system = ImprovedMovieRecommendationSystem(cache_dir_name="cached_features")
        recommender_system.prepare_data(movies_dataframe)

        logger.info("✅ 로컬 추천 시스템 초기화 및 데이터 전처리 완료.")

    except Exception as e:
        traceback.print_exc()
        logger.error(f"❌ 로컬 추천 시스템 초기화 중 오류 발생: {e}")
        raise RuntimeError(f"로컬 추천 시스템 초기화 실패: {e}")

def load_vectors_from_s3(bucket_name: str, key: str) -> np.ndarray:
    """S3에서 NPY 파일을 빠르게 로드"""
    try:
        logger.info(f"S3에서 NPY 로드: s3://{bucket_name}/{key}")
        
        response = s3.get_object(Bucket=bucket_name, Key=key)
        npy_data = io.BytesIO(response['Body'].read())
        vectors = np.load(npy_data)
        
        logger.info(f"NPY 로드 완료: {vectors.shape}, {vectors.nbytes / 1024**2:.2f} MB")
        return vectors
        
    except Exception as e:
        logger.error(f"NPY 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"NPY 로드 실패: {e}")
    
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """DataFrame의 메모리 사용량을 최적화"""
    logger.info("DataFrame 메모리 최적화 시작...")
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"최적화 전 메모리 사용량: {initial_memory:.2f} MB")
    
    # 숫자형 컬럼 최적화
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # 문자열 컬럼을 category로 변환 (중복이 많은 경우)
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # 유니크 값이 50% 미만인 경우
            df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"최적화 후 메모리 사용량: {final_memory:.2f} MB")
    logger.info(f"메모리 절약: {initial_memory - final_memory:.2f} MB ({(initial_memory - final_memory) / initial_memory * 100:.1f}%)")
    
    return df
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("메인 FastAPI 애플 시작: 모든 추천 시스템 데이터 초기화 시작.")

    """서버 시작 시 불변 데이터 미리 로드"""
    global cached_movie_df, cached_features, cached_movie_vectors
    
    print("🚀 영화 데이터 캐싱 시작...")
    
    try:
        # # S3에서 CSV 로드 (청크 크기 조정 가능)
        # csv_key = 'popcorithm_contents_metadata.csv'
        # print("📊 CSV 파일 로딩 중...")
        # cached_movie_df = load_csv_from_s3(S3_BUCKET_NAME, csv_key, chunksize=5000)  # 청크 크기를 5000으로 설정
        
        # # DataFrame 메모리 최적화
        # cached_movie_df = optimize_dataframe_memory(cached_movie_df)
        # print(f"✅ S3 CSV 로드 및 최적화 완료: {len(cached_movie_df)}개 영화")
        
        # # S3에서 JSON 로드
        # json_key = 'popcorithm_with_features.json'
        # print("📋 JSON 파일 로딩 중...")
        # cached_features = load_json_from_s3(S3_BUCKET_NAME, json_key, check_size=True)
        # print(f"✅ S3 JSON 로드 완료: {len(cached_features['actors'])}명 배우")
        
        # # 벡터 미리 변환 (메모리 효율적으로)
        # print("🔢 벡터 로드 중...")
        # cached_movie_vectors = load_vectors_from_s3(S3_BUCKET_NAME, 'movie_vectors.npy')
        
        # # 가비지 컬렉션으로 메모리 정리
        # gc.collect()
        
        # print("🎉 S3 캐싱 완료! 추천 API 준비됨")
        
        # # data_loader의 load_all_data 함수를 직접 호출
        # load_all_data()
        load_all_filtering_data()
        await initialize_local_recommender_system()

        logger.info("추천 시스템 데이터 초기화 완료.")
        yield
        
    except Exception as e:
        logger.error(f"추천 시스템 초기화 오류 발생: {e}", exc_info=True)
        # 메모리 정리
        gc.collect()
        raise RuntimeError("추천 시스템 초기화 실패") from e

# Router 를 통해 메인에다 연결
# CORS 설정 #
app = FastAPI(title="Popco Recommender API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://popco.site", "http://www.popco.site"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["Authorization", "Cache-Control", "Content-Type"],
)

app.include_router(persona_recommender_router)
app.include_router(filtering_recommender_router, prefix="/recommends/filters")

@app.get("/", tags=["기본"])
def root():
    return {
        "message": "Popco 추천 시스템 API",
        "status": "running",
        "docker_env": os.getenv("DOCKER_ENV") is not None
    }

@app.get("/health", tags=["시스템"])
def health_check():
    """시스템의 전반적인 상태를 확인합니다."""
    try:
        # 데이터베이스 연결 상태 확인
        db_connected = check_db_connection()
        
        return {
            "status": "healthy",
            "database": "connected" if db_connected else "disconnected"
        }
        
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"시스템 상태 확인 중 오류 발생: {e}")

@app.get("/debug/data-sizes", tags=["관리자"])
def check_data_sizes():
    """캐시된 데이터 크기 확인, 디버깅 용도"""
    return {
        "movie_df_shape": cached_movie_df.shape if cached_movie_df is not None else "None",
        "movie_vectors_shape": cached_movie_vectors.shape if cached_movie_vectors is not None else "None", 
        "features_actors_count": len(cached_features['actors']) if cached_features else "None",
        "features_total_size": len(cached_features['genres']) + len(cached_features['actors']) + len(cached_features['directors']) if cached_features else "None"
    }

@app.post("/admin/generate-vectors-npy", tags=["관리자"])
def generate_vectors_npy():
    """관리자가 벡터를 numpy 바이너리 파일로 미리 생성하여 S3에 업로드, 프론트에서 사용하지 않습니다."""
    try:
        print("🔢 벡터 NPY 파일 생성 시작...")
        
        # 1. CSV에서 벡터 데이터 로드
        csv_key = 'popcorithm_with_vectors.csv'
        df = load_csv_from_s3(S3_BUCKET_NAME, csv_key, chunksize=5000)
        
        # 2. 벡터 변환 (기존 방식)
        print("벡터 변환 중...")
        movie_vectors = np.array([
            list(map(float, row['vector'].split(',')))
            for _, row in df.iterrows()
        ], dtype=np.float32)
        
        # 3. NPY 파일로 S3 업로드
        npy_buffer = io.BytesIO()
        np.save(npy_buffer, movie_vectors)
        npy_buffer.seek(0)
        
        s3.put_object(
            Bucket=S3_BUCKET_NAME,
            Key='movie_vectors.npy',
            Body=npy_buffer.getvalue()
        )
        
        print(f"✅ NPY 파일 업로드 완료: {movie_vectors.shape}")
        
        return {
            "message": "벡터 NPY 파일 생성 및 S3 업로드 완료",
            "shape": movie_vectors.shape,
            "file_size_mb": movie_vectors.nbytes / 1024**2
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NPY 생성 실패: {e}")
    
@app.post("/admin/popcorithms/create-csv", tags=["관리자"])
def init_popcorithm_csv():
    """관리자가 팝코리즘 벡터 csv 를 생성하는 API 입니다. 프론트에서 사용하지 않습니다."""
    try:
        create_movie_metadata_csv()
        return {"message": "서버에 movie_metadata.csv 파일 생성을 성공적으로 완료했습니다."}
    except Exception as e:
        traceback.print_exc()  # 자바의 printStackTrace와 동일
        
        # 스택 트레이스를 문자열로 가져오기
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"파일 생성 중 오류 발생: {str(e)}")
    
@app.post("/admin/create-metadata-csv", tags=["관리자"])
def create_metadata_csv_api():
    """ 관리자가 순수 콘텐츠 메타데이터 csv 를 생성하는 API, 프론트에서 사용하지 않습니다. """
    try:
        print("📋 순수 메타데이터 CSV 생성 시작...")
        result_df = create_metadata_only_csv()
        
        return {
            "message": "순수 메타데이터 CSV 생성이 성공적으로 완료되었습니다.",
            "details": {
                "file_name": "popcorithm_contents_metadata.csv",
                "total_movies": len(result_df),
                "file_type": "metadata_only",
                "estimated_size": "~50MB",
                "generation_time": "빠름 (30초 내외)"
            }
        }
        
    except Exception as e:
        print(f"메타데이터 CSV 생성 오류: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500, 
            detail=f"메타데이터 CSV 생성 중 오류 발생: {str(e)}"
        )
    
@app.get("/recommends/popcorithms/users/{userId}/limits/{limit}", tags=["추천 알고리즘"])
def recommends_by_popcorithm(user_id: int, limit: int):
    """사용자가 팝코리즘 추천을 받기위해 호출되어야 하는 API 입니다. userID 와 limit를 넣어서 사용해주세요."""
    try:
        # 1단계: 사용자 활동 조회
        activities = get_user_recent_activities(user_id) # activites : List[Dict]
        
        # 2단계: 선호도 계산
        preferences = calculate_user_preferences(activities) # preferences : Dict
        
        # 3단계: 추천 계산  
        watched_movies = [activity['movie_id'] for activity in activities] 
        
        recommendations = calculate_recommendations(preferences, cached_movie_df, cached_features, cached_movie_vectors, watched_movies, limit) # 
        
        # 4단계: 응답 반환
        return RecommendationResponse(
            user_id=user_id,
            recommendations=recommendations,
            total_count=len(recommendations),
            generated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
                # 상세 에러 정보 출력
        import traceback
        print("=== 에러 발생! ===")
        print(f"에러 메시지: {str(e)}")
        print(f"에러 타입: {type(e).__name__}")
        print("\n=== 전체 스택 트레이스 ===")
        traceback.print_exc()  # 콘솔에 출력
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommends/contents", response_model=RecommendationListResponse, tags=["추천 시스템"])
async def recommend_movies_api(request: RecommendRequest, db: Session = Depends(get_db)):
    if recommender_system is None:
        raise HTTPException(status_code=503, detail="추천 시스템이 아직 초기화되지 않았습니다.")

    check_user_exists(db, request.user_id)

    try:
        result = get_top_ranked_content(db, batch_type=request.type)

        if result:
            content_id, content_type = result

            # 1. DB에서 기존 추천 데이터 확인
            existing_recommendations = get_existing_recommendations(
                db, content_id, content_type, request.user_id
            )
            
            # 2. DB에 데이터가 있으면 바로 반환
            if existing_recommendations:
                return RecommendationListResponse(recommendations=existing_recommendations)
            
            # 3. DB에 데이터가 없으면 추천 시스템 실행
            raw_recommendations = recommender_system.recommend_movies(
                content_id=content_id,
                content_type=content_type,
                top_n=8,
                use_adaptive_weights=True
            )

            if raw_recommendations is None:
                raise HTTPException(
                    status_code=404, 
                    detail=f"ID '{content_id}', Type '{content_type}'에 일치하는 영화를 찾을 수 없습니다."
                )

            # 4. dict를 RecommendationResponse로 변환
            recommendations = []
            for rec_dict in raw_recommendations:
                recommendation = ContentRecommendationListResponse(
                    content_id=rec_dict['content_id'],
                    content_type=rec_dict.get('content_type'),
                    title=rec_dict['title'],
                    total_similarity=rec_dict['total_similarity'],
                    poster_path=rec_dict['poster_path']
                )
                recommendations.append(recommendation)

            # 5. 추천 결과를 DB에 저장 (RecommendationResponse 객체들로)
            save_success = save_recommendations_to_db(
                db, content_id, content_type, recommendations
            )
            
            if not save_success:
                print("DB 저장에 실패했지만 추천 결과는 반환합니다.")

            # user_id가 있다면 전달
            return build_recommendation_responses(db, recommendations, user_id=request.user_id)
        
        else:
            # 인기 콘텐츠가 없는 경우
            raise HTTPException(status_code=404, detail="랭킹된 인기 콘텐츠가 없습니다.")
                    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"추천 생성 중 오류 발생: {str(e)}")
