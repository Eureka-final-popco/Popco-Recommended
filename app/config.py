from pydantic_settings import BaseSettings
import os, sys

class Settings(BaseSettings):
    DB_HOST: str
    DB_PORT: int
    DB_USERNAME: str
    DB_PASSWORD: str
    DB_NAME: str

    class Config:
        if os.getenv("DOCKER_ENV"):
            env_file = ".env"
        else:
            env_file = ".env.local"

settings = Settings()