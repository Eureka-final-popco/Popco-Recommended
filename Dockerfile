# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 및 Java 설치
RUN apt-get update && apt-get install -y build-essential default-jdk

# JAVA_HOME 환경 변수 설정
ENV JAVA_HOME /usr/lib/jvm/default-java

# 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# # 앱 코드 복사 - 볼륨 테스트
# COPY . .

# 포트 노출
EXPOSE 8000

# 서버 실행
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]