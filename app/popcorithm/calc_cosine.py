import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import pickle

def load_movie_metadata(csv_path: str) -> pd.DataFrame:
    """
    영화 메타데이터 CSV 파일 로드
    """
    df = pd.read_csv(csv_path)
    
    # 문자열을 리스트로 변환
    df['genres_list'] = df['genres'].apply(lambda x: x.split(',') if pd.notna(x) and x else [])
    df['actors_list'] = df['actors'].apply(lambda x: x.split(',') if pd.notna(x) and x else [])
    df['directors_list'] = df['directors'].apply(lambda x: x.split(',') if pd.notna(x) and x else [])
    
    return df

def load_movie_metadata_with_vectors(csv_path):
    """벡터 포함 영화 메타데이터 로드"""
    df = pd.read_csv(csv_path)
    
    # 벡터 컬럼을 numpy 배열로 변환
    movie_vectors = []
    for vector_str in df['vector']:
        # "1.0,0.0,1.0,..." → [1.0, 0.0, 1.0, ...]
        vector = [float(x) for x in vector_str.split(',')]
        movie_vectors.append(vector)
    
    movie_vectors = np.array(movie_vectors)
    
    return df, movie_vectors

def extract_all_features(movies_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    모든 영화에서 고유한 장르/배우/감독 추출
    """
    all_genres = set()
    all_actors = set()
    all_directors = set()
    
    for _, movie in movies_df.iterrows():
        all_genres.update([g.strip() for g in movie['genres_list'] if g.strip()])
        all_actors.update([a.strip() for a in movie['actors_list'] if a.strip()])
        all_directors.update([d.strip() for d in movie['directors_list'] if d.strip()])
    
    return {
        'genres': sorted(list(all_genres)),
        'actors': sorted(list(all_actors)), 
        'directors': sorted(list(all_directors))
    }

def create_movie_vectors(movies_df: pd.DataFrame, all_features: Dict[str, List[str]]) -> np.ndarray:
    """
    모든 영화를 벡터 형태로 변환
    각 영화는 [장르벡터, 배우벡터, 감독벡터]로 구성
    """
    movie_vectors = []
    
    print(f"영화 벡터 생성 중... (총 {len(movies_df)}개)")
    
    for idx, movie in movies_df.iterrows():
        movie_vector = []
        
        # 장르 벡터 (해당하면 1, 아니면 0)
        for genre in all_features['genres']:
            if genre in movie['genres_list']:
                movie_vector.append(1.0)
            else:
                movie_vector.append(0.0)
        
        # 배우 벡터 (해당하면 1, 아니면 0)
        for actor in all_features['actors']:
            if actor in movie['actors_list']:
                movie_vector.append(1.0)
            else:
                movie_vector.append(0.0)
        
        # 감독 벡터 (해당하면 1, 아니면 0)
        for director in all_features['directors']:
            if director in movie['directors_list']:
                movie_vector.append(1.0)
            else:
                movie_vector.append(0.0)
        
        movie_vectors.append(movie_vector)
        
        if (idx + 1) % 1000 == 0:
            print(f"  {idx + 1}개 영화 처리 완료...")
    
    return np.array(movie_vectors)

def calculate_recommendations(preferences, cached_movie_df, cached_features, cached_movie_vectors, watched_movies, limit=10):
    """
    사용자 선호도를 바탕으로 추천 영화 계산
    """
    print("1 + 2 : 캐싱된 데이터 로드 중...")
    movies_df = cached_movie_df
    all_features = cached_features
    movie_vectors = cached_movie_vectors
    
    print("3. 사용자 선호도 벡터 생성 중...")
    user_vector = create_user_preference_vector(preferences, all_features)
    
    # --- 디버깅 코드 추가 ---
    if np.all(user_vector == 0):
        print(f"경고: user_id 의 선호도 벡터가 제로 벡터입니다.")
        print(f"사용자 선호도: {preferences}")

    print("4. 코사인 유사도 계산 중...")
    user_vector_2d = user_vector.reshape(1, -1)
    similarities = cosine_similarity(user_vector_2d, movie_vectors)[0]  # ← 여기서 사용!
    
    print("5. 추천 결과 생성 중...")
    recommendations = []
    for idx, similarity in enumerate(similarities):
    
        # movies_df에서 해당 영화의 행(row)을 가져옴
        movie = movies_df.iloc[idx]
        movie_id = int(movie['movie_id'])
    
        # 사용자가 이미 본 영화는 추천 목록에서 제외
        if movie_id not in watched_movies:
        
            # Pydantic 모델(MovieRecommendation)의 필드에 맞게 딕셔너리 생성
            recommendations.append({
                'content_id': movie_id,
                'type': movie['type'],
                'title': movie['title'],
                'score': float(similarity), # 모델의 score 필드에 유사도 점수를 할당
                'poster_path': movie['poster_path'], 
                # .split() 사용 전, 값이 문자열(str)인지 확인하여 에러 방지
                'genres': movie['genres'].split(',') if isinstance(movie['genres'], str) else [],
                'main_actors': movie['actors'].split(',') if isinstance(movie['actors'], str) else [],
                'directors': movie['directors'].split(',') if isinstance(movie['directors'], str) else []
            })

    # 유사도 점수 기준으로 정렬
    # 'score' 키를 기준으로 정렬
    recommendations.sort(key=lambda x: x['score'], reverse=True)

    return recommendations[:limit]

def create_user_preference_vector(user_preferences: Dict, 
                                all_features: Dict[str, List[str]]) -> np.ndarray:
    """
    사용자 선호도를 벡터 형태로 변환 (이전 단계에서 정의한 함수)
    """
    preference_vector = []
    
    # 장르 점수 벡터화
    for genre in all_features['genres']:
        score = user_preferences['genres'].get(genre, 0.0)
        preference_vector.append(score)
    
    # 배우 점수 벡터화  
    for actor in all_features['actors']:
        score = user_preferences['actors'].get(actor, 0.0)
        preference_vector.append(score)
    
    # 감독 점수 벡터화
    for director in all_features['directors']:
        score = user_preferences['directors'].get(director, 0.0)
        preference_vector.append(score)
    
    return np.array(preference_vector)

# 전체 프로세스 테스트
def test_full_recommendation_process():
    """
    전체 추천 프로세스 테스트
    """
    # 1. 샘플 사용자 선호도 (이전 단계 결과)
    sample_preferences = {
        'genres': {'액션': 7.2, 'SF': 6.0, '모험': 3.6, '로맨스': -0.3},
        'actors': {'로버트 다우니 주니어': 8.0, '크리스 에반스': 3.6, '기네스 팰트로': 3.2},
        'directors': {'루소 형제': 3.6, '존 파브로': 2.4}
    }
    
    # 2. 샘플 영화 데이터 생성 (실제로는 CSV에서 로드)
    sample_movies = pd.DataFrame([
        {'movie_id': 1, 'title': '토르', 'genres': '액션,SF', 'actors': '크리스 헴스워스,나탈리 포트만', 'directors': '케네스 브래너'},
        {'movie_id': 2, 'title': '캡틴 아메리카', 'genres': '액션,모험', 'actors': '크리스 에반스,헤일리 앳웰', 'directors': '조 존스턴'},
        {'movie_id': 3, 'title': '타이타닉', 'genres': '로맨스,드라마', 'actors': '레오나르도 디카프리오,케이트 윈슬릿', 'directors': '제임스 카메론'},
        {'movie_id': 4, 'title': '가디언즈 오브 갤럭시', 'genres': '액션,SF,코미디', 'actors': '크리스 프랫,조 샐다나', 'directors': '제임스 건'}
    ])
    
    # 데이터 전처리
    sample_movies['genres_list'] = sample_movies['genres'].apply(lambda x: x.split(','))
    sample_movies['actors_list'] = sample_movies['actors'].apply(lambda x: x.split(','))
    sample_movies['directors_list'] = sample_movies['directors'].apply(lambda x: x.split(','))
    
    # 3. 사용자가 이미 본 영화 (추천에서 제외)
    watched_movies = []  # 테스트용으로 빈 리스트
    
    # 4. 추천 계산
    recommendations = calculate_recommendations(
        user_preferences=sample_preferences,
        movies_df=sample_movies,
        user_watched_movies=watched_movies,
        top_n=3
    )
    
    return recommendations

if __name__ == "__main__":
    test_recommendations = test_full_recommendation_process()