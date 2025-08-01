LIKE_WEIGHT = 5.0          # 좋아요 가중치 
DISLIKE_WEIGHT = -5.0      # 싫어요 가중치 
STAR_RATING_WEIGHTS = {    # 별점 가중치 
    0.5: -5.0,
    1.0: -3.5,
    1.5: -2.0,
    2.0: -1.0,
    2.5: 0.0,
    3.0: 0.5,
    3.5: 1.0,
    4.0: 2.0,
    4.5: 4.0,
    5.0: 5.0
}
MIN_INTERACTIONS_FOR_POPULARITY = 5 # 연령대별 인기 콘텐츠 계산을 위한 최소 상호작용 수
MIN_INTERACTIONS_PER_PERSONA_POPULAR = 3 # 페르소나별 인기 계산을 위한 최소 상호작용 수

RECOMMENDATION_COUNT=40