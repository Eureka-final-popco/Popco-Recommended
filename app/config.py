from pydantic_settings import BaseSettings
from typing import Optional
import os, sys
import logging

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
logger = logging.getLogger(__name__)

# 디버깅 로그
logger.info("=== 환경변수 디버깅 ===")
logger.info(f"MY_AWS_S3_BUCKET_NAME: {os.getenv('MY_AWS_S3_BUCKET_NAME')}")
logger.info(f"MY_AWS_ACCESS_KEY_ID: {os.getenv('MY_AWS_ACCESS_KEY_ID')}")
logger.info(f"DB_HOST: {os.getenv('DB_HOST')}")
logger.info(f"현재 디렉토리: {os.getcwd()}")
logger.info(f".env 파일 존재: {os.path.exists('.env')}")


class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_NAME: str

    MY_AWS_S3_BUCKET_NAME: str
    MY_AWS_S3_REGION: str
    MY_AWS_ACCESS_KEY_ID: str
    MY_AWS_SECRET_ACCESS_KEY: str

    ENVIRONMENT: Optional[str] = None
    DOCKER_HUB: Optional[str] = None
    IMAGE_TAG: Optional[str] = None
    
    class Config:
        # 환경 우선순위: ENVIRONMENT > DOCKER_ENV > 기본값
        env_file = ".local.env"  # 기본값
        
        # def __init__(self):
            # AWS 배포 환경 (최우선)
            # if os.getenv("ENVIRONMENT") == "production":
            #     self.env_file = ".env.prod"
            # # 로컬 도커 환경
            # elif os.getenv("DOCKER_ENV") == "true":
            #     self.env_file = ".env.docker"
            # # 로컬 개발 환경 (기본값)
            # else:
            #     self.env_file = ".env.local"

settings = Settings()