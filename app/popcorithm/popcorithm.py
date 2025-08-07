from collections import defaultdict
from typing import Dict, List
import numpy as np

def calculate_user_preferences(activities: List[Dict], 
                             genre_weight: float = 0.375,
                             actor_weight: float = 0.525, 
                             director_weight: float = 0.1) -> Dict:
    """
    사용자 활동 데이터로부터 선호도 프로필을 계산
    
    Args:
        activities: get_user_recent_activities()에서 반환된 활동 데이터
        genre_weight: 장르 가중치 (기본값 0.3)
        actor_weight: 배우 가중치 (기본값 0.4)
        director_weight: 감독 가중치 (기본값 0.3)
    
    Returns:
        사용자 선호도 프로필 딕셔너리
    """
    
    actor_weights = [0.35, 0.35, 0.1, 0.1, 0.1]
    genre_weights = [0.4, 0.3, 0.15, 0.15]
    
    # 각 카테고리별 누적 점수 저장
    genre_scores = defaultdict(float)
    actor_scores = defaultdict(float)
    director_scores = defaultdict(float)
    
    print(f"\n=== 사용자 선호도 계산 시작 (총 {len(activities)}건 활동) ===")
    
    for activity in activities:
        movie_score = activity['total_score']
        movie_title = activity['title']
        
        print(f"\n처리중: {movie_title} (점수: {movie_score})")
        
        # 장르별 점수 누적 (가중치 적용)
        for idx, genre in enumerate(activity['genres'][:4]):  # 최대 4개 장르
            if genre.strip():
                # 순서에 따른 가중치 계산
                genre_specific_weight = genre_weights[idx] if idx < len(genre_weights) else 0.05
                weighted_score = movie_score * genre_weight * genre_specific_weight
                genre_scores[genre.strip()] += weighted_score
                print(f"  장르 '{genre}' ({idx+1}순위): +{weighted_score:.2f}")

                # 배우별 차등 가중치 적용
        for idx, actor in enumerate(activity['main_actors'][:5]):  # 최대 5명
            if actor.strip():
                # 순서에 따른 가중치 계산
                actor_specific_weight = actor_weights[idx] if idx < len(actor_weights) else 0.02
                weighted_score = movie_score * actor_weight * actor_specific_weight
                actor_scores[actor.strip()] += weighted_score
                print(f"  배우 '{actor}' ({idx+1}순위): +{weighted_score:.2f}")
        
        # 감독별 점수 누적 (가중치 적용)
        for director in activity['directors']:
            if director.strip():  # 빈 문자열 제외
                weighted_score = movie_score * director_weight
                director_scores[director.strip()] += weighted_score
                print(f"  감독 '{director}': +{weighted_score:.2f}")
    
    # 결과 정리
    user_preferences = {
        'genres': dict(genre_scores),
        'actors': dict(actor_scores), 
        'directors': dict(director_scores),
        'weights': {
            'genre': genre_weight,
            'actor': actor_weight,
            'director': director_weight
        },
        'total_activities': len(activities)
    }
    
    # 결과 출력
    print(f"\n=== 최종 사용자 선호도 프로필 ===")
    
    print(f"\n[장르 선호도 TOP 5]")
    sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
    for genre, score in sorted_genres[:5]:
        print(f"  {genre}: {score:.2f}")
    
    print(f"\n[배우 선호도 TOP 5]")
    sorted_actors = sorted(actor_scores.items(), key=lambda x: x[1], reverse=True)
    for actor, score in sorted_actors[:5]:
        print(f"  {actor}: {score:.2f}")
    
    print(f"\n[감독 선호도 TOP 5]")
    sorted_directors = sorted(director_scores.items(), key=lambda x: x[1], reverse=True)
    for director, score in sorted_directors[:5]:
        print(f"  {director}: {score:.2f}")
    
    return user_preferences

def create_user_preference_vector(user_preferences: Dict, 
                                all_features: Dict[str, List[str]]) -> np.ndarray:
    """
    사용자 선호도를 벡터 형태로 변환 (코사인 유사도 계산용)
    
    Args:
        user_preferences: calculate_user_preferences()의 결과
        all_features: {'genres': [...], 'actors': [...], 'directors': [...]}
                     전체 영화 데이터에서 추출한 모든 고유 특성들
    
    Returns:
        사용자 선호도 벡터 (numpy array)
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

# 테스트 함수
def test_preference_calculation():
    # 샘플 활동 데이터 (실제로는 get_user_recent_activities()에서 가져옴)
    sample_activities = [
        {
            'movie_id': 1,
            'title': '어벤져스',
            'total_score': 12.0,  # 별점 4.5 + 좋아요 = (4.5*2) + 3
            'genres': ['액션', 'SF', '모험'],
            'main_actors': ['로버트 다우니 주니어', '크리스 에반스', '스칼릿 요한슨'],
            'directors': ['루소 형제']
        },
        {
            'movie_id': 2,
            'title': '아이언맨',
            'total_score': 8.0,
            'genres': ['액션', 'SF'],
            'main_actors': ['로버트 다우니 주니어', '기네스 팰트로'],
            'directors': ['존 파브로']
        },
        {
            'movie_id': 3,
            'title': '로맨스 영화',
            'total_score': -1.0,  # 별점 2.0 + 싫어요 = (2.0*2) - 5
            'genres': ['로맨스', '드라마'],
            'main_actors': ['배우A', '배우B'],
            'directors': ['감독A']
        }
    ]
    
    # 선호도 계산
    preferences = calculate_user_preferences(sample_activities)
    
    return preferences

if __name__ == "__main__":
    test_preferences = test_preference_calculation()