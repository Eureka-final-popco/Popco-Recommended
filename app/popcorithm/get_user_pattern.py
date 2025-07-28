import mysql.connector
import pandas as pd
from typing import List, Dict
from datetime import datetime
from ..config import Settings

def get_user_recent_activities(user_id: int, limit: int = 30) -> List[Dict]:
    """
    사용자의 최근 활동 30건을 가져와서 통합된 데이터로 반환
    reviews와 content_reactions를 통합해서 처리
    """
    settings = Settings()

    connection = mysql.connector.connect(
    host=settings.DB_HOST,
    port=settings.DB_PORT,
    database=settings.DB_NAME,
    user=settings.DB_USERNAME,  # settings.DB_USER -> settings.DB_USERNAME
    password=settings.DB_PASSWORD
)
    
    # 사용자 활동 통합 조회 쿼리
    query = """
    (
        -- 리뷰 데이터 (별점 + 텍스트)
        SELECT 
            r.content_id as movie_id,
            r.score as rating,
            NULL as reaction,  -- 리뷰에는 reaction 없음
            r.created_at,
            c.title,
            
            -- 장르 정보
            GROUP_CONCAT(DISTINCT g.name ORDER BY g.name) as genres,
            
            -- 주연배우 정보 (상위 5명)
            (SELECT GROUP_CONCAT(name ORDER BY cast_order SEPARATOR ',')
 FROM (
     SELECT a.name, cm.cast_order
     FROM cast_members cm 
     JOIN actors a ON cm.actor_id = a.id 
     WHERE cm.content_id = r.content_id AND cm.type = r.type 
     ORDER BY cm.cast_order ASC 
     LIMIT 5
 ) AS limited_actors
) as main_actors,
            
            -- 감독 정보
        (SELECT GROUP_CONCAT(name ORDER BY name SEPARATOR ',') FROM (
            SELECT crew.name FROM crews cr JOIN crew_members crew ON cr.crew_member_id = crew.id 
            WHERE cr.content_id = r.content_id AND cr.type = r.type AND cr.job = 'Director'
            ORDER BY crew.name ASC LIMIT 5
        ) AS limited_directors) as directors,
            
            'review' as activity_type
            
        FROM reviews r
        JOIN contents c ON r.content_id = c.id AND r.type = c.type
        LEFT JOIN content_genres cg ON c.id = cg.content_id AND c.type = cg.content_type
        LEFT JOIN genres g ON cg.genre_id = g.id
        WHERE r.user_id = %s 
          AND r.type = 'movie'
        GROUP BY r.review_id, r.content_id, r.score, r.created_at, c.title
    )
    
    UNION ALL
    
    (
        -- 콘텐츠 반응 데이터 (좋아요/싫어요)
        SELECT 
            cr.content_id as movie_id,
            NULL as rating,  -- 반응에는 별점 없음
            cr.reaction,
            cr.created_at,
            c.title,
            
            -- 장르 정보
            GROUP_CONCAT(DISTINCT g.name ORDER BY g.name) as genres,
            
            -- 주연배우 정보
        (SELECT GROUP_CONCAT(name ORDER BY cast_order SEPARATOR ',') FROM (
            SELECT a.name, cm.cast_order FROM cast_members cm JOIN actors a ON cm.actor_id = a.id 
            WHERE cm.content_id = cr.content_id AND cm.type = cr.type ORDER BY cm.cast_order ASC LIMIT 5
        ) AS limited_actors) as main_actors,
            
            -- 감독 정보
        (SELECT GROUP_CONCAT(name ORDER BY name SEPARATOR ',') FROM (
            SELECT crew.name FROM crews cr_sub JOIN crew_members crew ON cr_sub.crew_member_id = crew.id
            WHERE cr_sub.content_id = cr.content_id AND cr_sub.type = cr.type AND cr_sub.job = 'Director'
            ORDER BY crew.name ASC LIMIT 5
        ) AS limited_directors) as directors,
            
            'reaction' as activity_type
            
        FROM content_reactions cr
        JOIN contents c ON cr.content_id = c.id AND cr.type = c.type
        LEFT JOIN content_genres cg ON c.id = cg.content_id AND c.type = cg.content_type
        LEFT JOIN genres g ON cg.genre_id = g.id
        WHERE cr.user_id = %s 
          AND cr.type = 'movie'
        GROUP BY cr.content_reaction_id, cr.content_id, cr.reaction, cr.created_at, c.title
    )
    
    ORDER BY created_at DESC
    LIMIT %s
    """
    
    try:
        # 쿼리 실행
        df = pd.read_sql(query, connection, params=[user_id, user_id, limit])
        df['rating'] = df['rating'].where(pd.notna(df['rating']), None)

        # 데이터 변환
        activities = []
        for _, row in df.iterrows():
            print(f"Raw data: movie_id={row['movie_id']}, rating={row['rating']}, reaction={row['reaction']}")
            print(f"Rating type: {type(row['rating'])}, Is null: {pd.isna(row['rating'])}")
    
    # 점수 계산 전에 디버깅
            score = 0.0
            if row['rating'] is not None and not pd.isna(row['rating']):
                normalized_rating = (float(row['rating']) - 2.5) * 2
                score += normalized_rating
            else:
                score += 0.0
    
            if row['reaction'] == 'LIKE':
                print(f"좋아요 처리: +3")
                score += 3
            elif row['reaction'] == 'DISLIKE':
                print(f"싫어요 처리: -5") 
                score -= 5
    
            print(f"최종 점수: {score} (타입: {type(score)})")
            
            activity = {
                'movie_id': int(row['movie_id']),
                'title': row['title'],
                'rating': float(row['rating']) if row['rating'] is not None else None,
                'reaction': row['reaction'],
                'total_score': score,
                'genres': row['genres'].split(',') if row['genres'] else [],
                'main_actors': row['main_actors'].split(',') if row['main_actors'] else [],
                'directors': row['directors'].split(',') if row['directors'] else [],
                'created_at': row['created_at'],
                'activity_type': row['activity_type']
            }
            activities.append(activity)
        
        print(f"사용자 {user_id}의 최근 {len(activities)}건 활동을 조회했습니다.")
        return activities
        
    except Exception as e:
        print(f"사용자 활동 조회 오류: {e}")
        return []
        
    finally:
        connection.close()

# 테스트 함수
def test_user_activities():
    user_id = 1  # 테스트용 사용자 ID
    activities = get_user_recent_activities(user_id)
    
    print("\n=== 사용자 활동 샘플 ===")
    for i, activity in enumerate(activities[:3]):  # 상위 3개만 출력
        print(f"\n{i+1}. {activity['title']}")
        print(f"   점수: {activity['total_score']}")
        print(f"   장르: {', '.join(activity['genres'])}")
        print(f"   배우: {', '.join(activity['main_actors'][:2])}...")  # 상위 2명만
        print(f"   감독: {', '.join(activity['directors'])}")
        print(f"   활동유형: {activity['activity_type']}")

if __name__ == "__main__":
    test_user_activities()