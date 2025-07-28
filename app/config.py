from pydantic_settings import BaseSettings
import os, sys

class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_NAME: str

    AWS_S3_BUCKET_NAME: str
    AWS_S3_REGION: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    
    class Config:
        # 환경 우선순위: ENVIRONMENT > DOCKER_ENV > 기본값
        env_file = ".env.local"  # 기본값
        
        def __init__(self):
            # AWS 배포 환경 (최우선)
            if os.getenv("ENVIRONMENT") == "production":
                self.env_file = ".env.prod"
            # 로컬 도커 환경
            elif os.getenv("DOCKER_ENV") == "true":
                self.env_file = ".env.docker"
            # 로컬 개발 환경 (기본값)
            else:
                self.env_file = ".env.local"

settings = Settings()