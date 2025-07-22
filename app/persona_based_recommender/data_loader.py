import pandas as pd
import re
import os
from sklearn.metrics.pairwise import cosine_similarity

# 장르 한글 매핑 함수
def map_genres_to_korean(genres_str):
    if pd.isna(genres_str) or genres_str == '(no genres listed)':
        return ["(장르 없음)"]
    genres = re.split(r'[|,]', genres_str)
    genres = [g.strip() for g in genres if g.strip()]
    return list(set(genres))

# 제목에서 연도 정보 제거 및 클리닝
def clean_title(title):
    if isinstance(title, str):
        cleaned = re.sub(r'\s*\(\d{4}\)$', '', title).strip() 
        cleaned = re.sub(r'\s*\(.*\)\s*', '', cleaned) 
        cleaned = re.sub(r'[^a-zA-Z0-9가-힣\s]', ' ', cleaned) 
        cleaned = re.sub(r'\s+', ' ', cleaned).strip() 
        return cleaned.lower()
    return title

# 데이터 로드하고 전처리하여 반환
def load_initial_data(data_path: str):
    print(f"데이터를 '{data_path}'에서 로드 중입니다...")

    # 1. content_data.csv 파일 로드 및 전처리
    contents_df = pd.read_csv(os.path.join(data_path, 'content_data.csv'))
    contents_df_persona = contents_df[['id', 'title', 'genres']].copy()
    contents_df_persona.rename(columns={'id': 'contentId'}, inplace=True)
    contents_df_persona['genres'] = contents_df_persona['genres'].apply(map_genres_to_korean)
    contents_df_persona['title_cleaned'] = contents_df_persona['title'].apply(clean_title)

    # 2. links.csv 로드 및 매핑 준비
    links_df = pd.read_csv(os.path.join(data_path, 'links.csv'))
    links_df.dropna(subset=['tmdbId'], inplace=True)
    links_df['tmdbId'] = links_df['tmdbId'].astype(int)

    # 3. ratings.csv (별점 데이터) 로드 및 contentId 매핑
    reviews_df = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
    reviews_df.rename(columns={'rating': 'score'}, inplace=True)
    reviews_df = pd.merge(reviews_df, links_df[['movieId', 'tmdbId']], on='movieId', how='inner')
    reviews_df.drop(columns=['movieId', 'timestamp'], inplace=True)
    reviews_df.rename(columns={'tmdbId': 'contentId'}, inplace=True)
    reviews_df = reviews_df[['userId', 'contentId', 'score']]
    valid_content_ids_from_content_data = contents_df_persona['contentId'].unique()
    reviews_df = reviews_df[reviews_df['contentId'].isin(valid_content_ids_from_content_data)].copy()


    # 4. content_reactions.csv (좋아요/싫어요 데이터) 로드 및 contentId 매핑
    content_reactions_df_raw = pd.read_csv(os.path.join(data_path, 'content_reactions.csv'))
    content_reactions_df_raw.rename(columns={'user_id': 'userId', 'content_id': 'contentId'}, inplace=True)
    content_reactions_df_raw = content_reactions_df_raw[['userId', 'contentId', 'reaction']]

    # content_reactions_df_raw에 있는 contentId가 movieId에 해당한다고 가정하고 links_df와 병합
    temp_reactions_df_for_merge = content_reactions_df_raw.rename(columns={'contentId': 'movieId'})
    content_reactions_df = pd.merge(temp_reactions_df_for_merge, links_df[['movieId', 'tmdbId']], on='movieId', how='inner')
    content_reactions_df.drop(columns=['movieId'], inplace=True)
    content_reactions_df.rename(columns={'tmdbId': 'contentId'}, inplace=True)
    content_reactions_df = content_reactions_df[['userId', 'contentId', 'reaction']]
    content_reactions_df = content_reactions_df[content_reactions_df['contentId'].isin(valid_content_ids_from_content_data)].copy()

    # 5. 콘텐츠 유사도 매트릭스 생성 (장르 기반)
    all_genres = sorted(list(set(genre for sublist in contents_df_persona['genres'] for genre in sublist)))
    genre_matrix = pd.DataFrame(0, index=contents_df_persona['contentId'], columns=all_genres)
    for index, row in contents_df_persona.iterrows():
        content_id = row['contentId']
        for genre in row['genres']:
            if genre in genre_matrix.columns:
                genre_matrix.loc[content_id, genre] = 1

    content_similarity_matrix = cosine_similarity(genre_matrix)
    content_similarity_df = pd.DataFrame(content_similarity_matrix,
                                         index=contents_df_persona['contentId'],
                                         columns=contents_df_persona['contentId'])

    print("데이터 로딩 및 초기 전처리 완료.")
    return contents_df_persona, content_similarity_df, reviews_df, content_reactions_df

# 사용자 페르소나 데이터 로드
def load_user_personas(user_data_path: str):
    if os.path.exists(user_data_path):
        print(f"'{user_data_path}'에서 사용자 페르소나를 로드 중입니다...")
        
        try:
            user_persona_df = pd.read_csv(user_data_path)
            
            # 'user_id' 컬럼을 숫자로 변환 시도, 오류 발생 시 NaN으로 처리
            user_persona_df['user_id'] = pd.to_numeric(user_persona_df['user_id'], errors='coerce')
            
            # user_id가 NaN인 행(유효하지 않은 user_id)을 제거
            initial_rows = len(user_persona_df)
            user_persona_df.dropna(subset=['user_id'], inplace=True)
            rows_after_dropna = len(user_persona_df)

            if initial_rows != rows_after_dropna:
                print(f"[경고] '{user_data_path}'에서 유효하지 않은 'user_id'가 포함된 {initial_rows - rows_after_dropna}개의 행을 제거했습니다.")
            
            # 남은 'user_id'를 정수형으로 변환 (이제 NaN 없음 보장)
            user_persona_df['user_id'] = user_persona_df['user_id'].astype(int)

            if not user_persona_df.empty:
                print(f"{len(user_persona_df['user_id'].unique())}명의 고유 사용자 페르소나 데이터를 로드했습니다.")
            else:
                print(f"[정보] '{user_data_path}' 파일이 비어 있거나 유효한 사용자 페르소나 데이터가 없습니다.")
            
            return user_persona_df
            
        except Exception as e:
            print(f"[오류] 사용자 페르소나 파일 '{user_data_path}' 로드 및 처리 중 오류 발생: {e}")
            print("파일 내용을 확인하거나, 파일을 삭제하고 다시 시도해보세요.")
            return pd.DataFrame(columns=['user_id', 'persona_id', 'score']) 
    else:
        print(f"[경고] 사용자 페르소나 파일 '{user_data_path}'를 찾을 수 없습니다. 빈 데이터프레임을 반환합니다.")
        return pd.DataFrame(columns=['user_id', 'persona_id', 'score'])


# 사용자 페르소나 데이터 csv 파일로 저장
def save_user_personas(user_persona_df, user_data_path: str):
    if not user_persona_df.empty:
        os.makedirs(os.path.dirname(user_data_path), exist_ok=True)
        user_persona_df.to_csv(user_data_path, index=False)
        print(f"사용자 페르소나 데이터가 '{user_data_path}'에 저장되었습니다.")
    else:
        print("[경고] 사용자 페르소나 데이터프레임이 비어 있어 저장하지 않습니다.")