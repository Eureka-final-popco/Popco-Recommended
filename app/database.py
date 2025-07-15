import pymysql
import pandas as pd
from config import settings

  # MySQL 서버에 연결
conn = pymysql.connect(
    host= settings.DB_HOST,         
    port= settings.DB_PORT, 
    user=settings.DB_USERNAME,
    password=settings.DB_PASSWORD,
    db=settings.DB_NAME,
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

try:
# 커서 생성
    with conn.cursor() as cursor:

        # 레코드 조회 쿼리 실행
        select_query = '''SELECT
                            c.id,
                            c.overview,
                            c.title,
                            c.release_date,
                            c.type,
                            GROUP_CONCAT(g.name ORDER BY g.name SEPARATOR ', ') AS `genres`
                        FROM `content` c
                        JOIN `content_genre_ids` cg ON c.id = cg.content_id
                        JOIN `genre` g ON cg.genre_id = g.id
                        GROUP BY 
                            c.id, 
                            c.overview,
                            c.title,
                            c.release_date,
                            c.type;'''
        cursor.execute(select_query)
        result = cursor.fetchall()

        # 조회 결과를 DataFrame으로 변환
        df = pd.DataFrame(result)
        df = pd.DataFrame(result, columns=['id', 'overview', 'title', 'release_date', 'genres','type'])

        # DataFrame 출력
        print(df[:10])
        df.to_csv('/data_processing/content_data.csv', index=False, encoding='utf-8-sig')

finally:
    # 연결 닫기
    conn.close()