# 상호작용 가중치 (user-item matrix 생성 시 사용)
LIKE_WEIGHT = 5.0          # 좋아요 가중치 (사용자 제공)
DISLIKE_WEIGHT = -5.0      # 싫어요 가중치 (사용자 제공)
STAR_RATING_WEIGHTS = {    # 별점 가중치 (사용자 제공)
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

# 피드백 및 페르소나 결정 관련 임계값
POSITIVE_FEEDBACK_THRESHOLD = 0.1               # 긍정적 피드백 임계값 (사용자 제공)
QA_WEIGHT = 0.5                                 # QA 답변 가중치 (사용자 제공)
CONTENT_FEEDBACK_WEIGHT = 0.5                   # 콘텐츠 피드백 가중치 (사용자 제공)
MIN_FEEDBACK_FOR_PERSONA_DETERMINATION = 1      # 페르소나 결정에 필요한 최소 피드백 수 (사용자 제공)
MIN_POSITIVE_RATINGS_FOR_GROWTH = 1             # 페르소나 성장을 위한 최소 긍정 평가 수 (사용자 제공)
PENALTY_FOR_NO_PERSONA_MATCH = -1.5             # 매칭되는 페르소나가 없을 때의 페널티 (사용자 제공)

# 유사도 임계값d
PERSONA_SIMILARITY_THRESHOLD = 0.5  # 페르소나 유사도 임계값 (추가됨)
CONTENT_SIMILARITY_THRESHOLD = 0.3  # 콘텐츠 유사도 임계값 (추가됨)
CF_SIMILARITY_THRESHOLD = 0.1       # 협업 필터링 유사도 임계값 (사용자 제공 및 일치)

# 추천 결과 수
TOP_N_CONTENTS_FOR_POPULAR_PER_PERSONA = 100    # 페르소나별 인기 콘텐츠 추천 시 가져올 상위 N개 (추가됨)
RECOMMENDATION_COUNT = 10                       # 최종 사용자에게 제공할 추천 콘텐츠 수 (사용자 제공 및 일치)
SIMILAR_USERS_COUNT = 50                        # 협업 필터링에서 유사 사용자를 찾을 때 고려할 수 (추가됨)
SIMILAR_CONTENTS_COUNT = 50                     # 콘텐츠 기반 필터링에서 유사 콘텐츠를 찾을 때 고려할 수 (추가됨)

# 인기 콘텐츠 관련
MIN_VOTE_COUNT_FOR_POPULARITY = 1               # 인기도 계산을 위한 최소 투표 수 (사용자 제공)

# QA 답변 기반 초기 추천을 위한 임계값
QA_INITIAL_RECOMMENDATION_THRESHOLD = 0.5       # QA 답변 기반 초기 추천을 위한 임계값 (사용자 제공)

# 페르소나 점수 관련
MAX_EXPECTED_PERSONA_SCORE = 100.0              # 임의의 최대 예상 점수 (사용자 제공)
MIN_PERSONA_SCORE = 0.1                         
BABY_PERSONA_THRESHOLD = 10.0                   # 메인 페르소나와 서브 페르소나 점수 차이 임계값 (사용자 제공)

# 캐시 설정 (초 단위)
RECOMMENDATION_CACHE_EXPIRATION_SECONDS = 3600 * 24 # 24시간 (데이터 새로고침 주기와 연관) (추가됨)

# 로깅 설정
LOG_LEVEL = "INFO" # DEBUG, INFO, WARNING, ERROR, CRITICAL (추가됨)
