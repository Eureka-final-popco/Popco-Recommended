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

from .config import settings
from .database import check_db_connection, get_db
from .popcorithm.models import (
    MovieRecommendation, 
    RecommendationResponse
)
from .popcorithm.create_popcorithm_csv import create_movie_metadata_csv
from .popcorithm.get_user_pattern import get_user_recent_activities
from .popcorithm.popcorithm import calculate_user_preferences
from .popcorithm.calc_cosine import calculate_recommendations, load_movie_metadata_with_vectors

from app.persona_based_recommender.state import pbr_app_state
from app.persona_based_recommender.persona_router import persona_recommender_router
from app.persona_based_recommender.data_loader import load_all_data

APP_ROOT_DIR = Path(__file__).parent # '/app'
S3_BUCKET_NAME = settings.MY_AWS_S3_BUCKET_NAME
s3 = boto3.client(
    's3',
    aws_access_key_id=settings.MY_AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.MY_AWS_SECRET_ACCESS_KEY,
    region_name=settings.MY_AWS_S3_REGION
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

cached_movie_df = None
cached_features = None
cached_movie_vectors = None

def load_csv_from_s3(bucket_name: str, key: str) -> pd.DataFrame:
    """S3에서 CSV 파일을 읽어서 DataFrame으로 반환"""
    try:
        logger.info(f"S3에서 CSV 로드 시작: s3://{bucket_name}/{key}")
        
        # S3에서 파일 객체 가져오기
        response = s3.get_object(Bucket=bucket_name, Key=key)
        
        # 바이너리 데이터를 pandas로 읽기
        csv_data = response['Body'].read()
        df = pd.read_csv(io.BytesIO(csv_data))
        
        logger.info(f"S3 CSV 로드 성공: {len(df)}개 행")
        return df
        
    except Exception as e:
        logger.error(f"S3 CSV 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"S3에서 CSV 로드 실패: {e}")

def load_json_from_s3(bucket_name: str, key: str) -> dict:
    """S3에서 JSON 파일을 읽어서 dict로 반환"""
    try:
        logger.info(f"S3에서 JSON 로드 시작: s3://{bucket_name}/{key}")
        
        # S3에서 파일 객체 가져오기
        response = s3.get_object(Bucket=bucket_name, Key=key)
        
        # JSON 데이터 파싱
        json_data = json.loads(response['Body'].read().decode('utf-8'))
        
        logger.info(f"S3 JSON 로드 성공")
        return json_data
        
    except Exception as e:
        logger.error(f"S3 JSON 로드 실패: {e}")
        raise HTTPException(status_code=500, detail=f"S3에서 JSON 로드 실패: {e}")
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("메인 FastAPI 애플 시작: 모든 추천 시스템 데이터 초기화 시작.")

    """서버 시작 시 불변 데이터 미리 로드"""
    global cached_movie_df, cached_features, cached_movie_vectors
    
    print("🚀 영화 데이터 캐싱 시작...")
    
    try:
        # S3에서 CSV 로드
        csv_key = 'popcorithm_with_vectors.csv'  # S3 키 경로
        cached_movie_df = load_csv_from_s3(S3_BUCKET_NAME, csv_key)
        print(f"✅ S3 CSV 로드 완료: {len(cached_movie_df)}개 영화")
        
        # S3에서 JSON 로드
        json_key = 'popcorithm_with_features.json'  # S3 키 경로
        cached_features = load_json_from_s3(S3_BUCKET_NAME, json_key)
        print(f"✅ S3 JSON 로드 완료: {len(cached_features['actors'])}명 배우")
        
        # 벡터 미리 변환
        cached_movie_vectors = np.array([
            list(map(float, row['vector'].split(',')))
            for _, row in cached_movie_df.iterrows()
        ], dtype=np.float32)
        print(f"✅ 벡터 변환 완료: {cached_movie_vectors.shape}")
        
        print("🎉 S3 캐싱 완료! 추천 API 준비됨")
        
        # data_loader의 load_all_data 함수를 직접 호출
        load_all_data() 
        logger.info("추천 시스템 데이터 초기화 완료.")
        yield
        
    except Exception as e:
        logger.error(f"추천 시스템 초기화 오류 발생: {e}", exc_info=True)
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

@app.post("/recommends/popcorithms/create-csv", tags=["관리자"])
def init_popcorithm_csv():
    """관리자가 팝코리즘 csv 를 생성하는 API 입니다. 프론트에서 사용하지 않습니다."""
    try:
        create_movie_metadata_csv()
        return {"message": "서버에 movie_metadata.csv 파일 생성을 성공적으로 완료했습니다."}
    except Exception as e:
        traceback.print_exc()  # 자바의 printStackTrace와 동일
        
        # 스택 트레이스를 문자열로 가져오기
        error_details = traceback.format_exc()
        raise HTTPException(status_code=500, detail=f"파일 생성 중 오류 발생: {str(e)}")
        
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
