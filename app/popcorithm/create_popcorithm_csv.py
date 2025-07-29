import pymysql
import pandas as pd
import json
from config import Settings
import traceback
from collections import defaultdict

def extract_all_features(df):
    """모든 영화에서 고유한 장르/배우/감독 추출 (배우/감독 수 제한)"""
    all_genres = set()
    actor_count = defaultdict(int)
    director_count = defaultdict(int)
    
    # 1단계: 모든 특성 수집 및 빈도 계산
    for _, row in df.iterrows():
        # 장르 추출 (모든 장르 유지)
        if row['genres']:
            all_genres.update([g.strip() for g in row['genres'].split(',') if g.strip()])
        
        # 배우 빈도 계산
        if row['actors']:
            for actor in row['actors'].split(','):
                if actor.strip():
                    actor_count[actor.strip()] += 1
        
        # 감독 빈도 계산
        if row['directors']:
            for director in row['directors'].split(','):
                if director.strip():
                    director_count[director.strip()] += 1
    
    # 2단계: 상위 N명만 선택
    top_actors = sorted(actor_count.items(), key=lambda x: x[1], reverse=True)[:15000]  # 상위 1000명
    top_directors = sorted(director_count.items(), key=lambda x: x[1], reverse=True)[:1000]  # 상위 500명
    
    print(f"\n=== 특성 제한 결과 ===")
    print(f"전체 배우: {len(actor_count)}명 → 상위 25000명 선택")
    print(f"전체 감독: {len(director_count)}명 → 상위 1000명 선택")
    print(f"상위 배우 예시: {[actor for actor, count in top_actors[:5]]}")
    print(f"상위 감독 예시: {[director for director, count in top_directors[:5]]}")
    
    return {
        'genres': sorted(list(all_genres)),
        'actors': [actor for actor, count in top_actors],
        'directors': [director for director, count in top_directors]
    }

def create_movie_vector(movie_row, all_features):
    """단일 영화를 벡터로 변환"""
    vector = []
    
    # 장르 벡터 (해당하면 1, 아니면 0)
    movie_genres = [g.strip() for g in movie_row['genres'].split(',') if g.strip()] if movie_row['genres'] else []
    for genre in all_features['genres']:
        vector.append(1.0 if genre in movie_genres else 0.0)
    
    # 배우 벡터 (해당하면 1, 아니면 0)
    movie_actors = [a.strip() for a in movie_row['actors'].split(',') if a.strip()] if movie_row['actors'] else []
    for actor in all_features['actors']:
        vector.append(1.0 if actor in movie_actors else 0.0)
    
    # 감독 벡터 (해당하면 1, 아니면 0)
    movie_directors = [d.strip() for d in movie_row['directors'].split(',') if d.strip()] if movie_row['directors'] else []
    for director in all_features['directors']:
        vector.append(1.0 if director in movie_directors else 0.0)
    
    return vector

def create_movie_metadata_csv():
    """
    DB에서 모든 영화 메타데이터를 가져와서 최적화된 CSV로 저장
    """
    settings = Settings()

    connection = pymysql.connect(
        host=settings.DB_HOST,
        port=settings.DB_PORT,
        database=settings.DB_NAME,
        user=settings.DB_USERNAME,
        password=settings.DB_PASSWORD,
        charset='utf8mb4'
    )
    
    # 영화 메타데이터 추출 쿼리
    query = """
    SELECT 
        c.id as movie_id,
        c.title,
        c.type,
        c.poster_path,
        
        -- 장르 정보 (JSON 형태로 집계)
        COALESCE(
            JSON_ARRAYAGG(
                CASE WHEN g.name IS NOT NULL 
                THEN g.name 
                ELSE NULL END
            ), 
            JSON_ARRAY()
        ) as genres,
        
        -- 주연배우 정보 (상위 5명, JSON 형태)
        (SELECT COALESCE(
            JSON_ARRAYAGG(a.name), 
            JSON_ARRAY()
        )
         FROM cast_members cm 
         JOIN actors a ON cm.actor_id = a.id 
         WHERE cm.content_id = c.id 
           AND cm.type = c.type 
         ORDER BY cm.cast_order ASC 
         LIMIT 5
        ) as main_actors,
        
        -- 감독 정보 (JSON 형태)
        (SELECT COALESCE(
            JSON_ARRAYAGG(crew.name), 
            JSON_ARRAY()
        )
         FROM crews cr 
         JOIN crew_members crew ON cr.crew_member_id = crew.id 
         WHERE cr.content_id = c.id 
           AND cr.type = c.type 
           AND cr.job = 'Director'
           LIMIT 1
        ) as directors
        
    FROM contents c
    LEFT JOIN content_genres cg ON c.id = cg.content_id AND c.type = cg.content_type
    LEFT JOIN genres g ON cg.genre_id = g.id
    GROUP BY c.id, c.title, c.type
    ORDER BY c.id
    """
    
    try:
        print("1단계: DB에서 영화 메타데이터 조회 중...")
        df = pd.read_sql(query, connection)
        
        # JSON 문자열을 파이썬 리스트로 변환
        def parse_json_column(json_str):
            if pd.isna(json_str):
                return []
            try:
                return json.loads(json_str)
            except:
                return []
        
        df['genres'] = df['genres'].apply(parse_json_column)
        df['main_actors'] = df['main_actors'].apply(parse_json_column)
        df['directors'] = df['directors'].apply(parse_json_column)
        
        # 리스트를 문자열로 변환 (CSV 저장용)
        df['genres_str'] = df['genres'].apply(lambda x: ','.join([genre for genre in x if genre is not None]) if x else '')
        df['actors_str'] = df['main_actors'].apply(lambda x: ','.join(x) if x else '')
        df['directors_str'] = df['directors'].apply(lambda x: ','.join(x) if x else '')
        
        # 기본 CSV용 컬럼 선택
        csv_df = df[['movie_id', 'type', 'title', 'poster_path', 'genres_str', 'actors_str', 'directors_str']].copy()
        csv_df.columns = ['movie_id', 'type', 'title', 'poster_path', 'genres', 'actors', 'directors']
        
        print(f"2단계: {len(csv_df)}개 영화 메타데이터 추출 완료")
        
        # ===== 최적화된 벡터화 로직 =====
        print("3단계: 고유 특성 추출 및 상위 N명 선택 중...")
        all_features = extract_all_features(csv_df)
        
        print(f"\n=== 최종 벡터 구성 ===")
        print(f"  - 장르: {len(all_features['genres'])}개")
        print(f"  - 배우: {len(all_features['actors'])}개 (상위 1000명)")
        print(f"  - 감독: {len(all_features['directors'])}개 (상위 500명)")
        
        total_vector_size = len(all_features['genres']) + len(all_features['actors']) + len(all_features['directors'])
        estimated_memory_mb = (len(csv_df) * total_vector_size * 8) / (1024 * 1024)
        
        print(f"  - 총 벡터 크기: {total_vector_size}차원")
        print(f"  - 예상 메모리 사용량: {estimated_memory_mb:.1f}MB")
        
        if estimated_memory_mb > 2000:
            print("⚠️  메모리 사용량이 2GB를 초과합니다!")
        else:
            print("✅ 메모리 사용량이 안전한 범위입니다.")
        
        print("\n4단계: 영화 벡터 생성 중...")
        movie_vectors = []
        
        for idx, movie in csv_df.iterrows():
            vector = create_movie_vector(movie, all_features)
            # 벡터를 문자열로 저장 (CSV용)
            vector_str = ','.join(map(str, vector))
            movie_vectors.append(vector_str)
            
            # 진행상황 출력 (더 자주)
            if (idx + 1) % 500 == 0:
                progress = ((idx + 1) / len(csv_df)) * 100
                print(f"  - {idx + 1}/{len(csv_df)} 영화 벡터 생성 완료... ({progress:.1f}%)")
        
        # 벡터 컬럼 추가
        csv_df['vector'] = movie_vectors
        
        print("5단계: 파일 저장 중...")
        
        # 벡터 포함 CSV 저장
        csv_filename = 'popcorithm_with_vectors.csv'
        csv_df.to_csv(csv_filename, index=False, encoding='utf-8')
        
        # 최적화된 특성 리스트 JSON 저장
        features_filename = 'popcorithm_with_features.json'
        with open(features_filename, 'w', encoding='utf-8') as f:
            json.dump(all_features, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== 최적화된 생성 완료 ===")
        print(f"총 {len(csv_df)}개 영화 데이터를 최적화하여 저장했습니다.") ##
        print(f"CSV 파일: {csv_filename}")
        print(f"특성 파일: {features_filename}")
        
        print("\n샘플 데이터:")
        print(csv_df[['movie_id','type', 'title', 'poster_path', 'genres', 'actors', 'directors']].head())
        
        return csv_df
        
    except Exception as e:
        print(f"오류 발생: {e}")
        traceback.print_exc()
        error_details = traceback.format_exc()
        print(f"\n문자열 형태:\n{error_details}")
        raise e
        
    finally:
        connection.close()

# 실행
if __name__ == "__main__":
    movie_df = create_movie_metadata_csv()