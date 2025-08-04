# 키워드 추천 개선
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import re
import os, pickle, json
from collections import Counter
from konlpy.tag import Okt

class ImprovedMovieRecommendationSystem:
    def __init__(self, stopword_files=None, cache_dir_name="cached_features"):
        self.kobert_tokenizer = None
        self.kobert_model = None
        self.tfidf_title = None
        self.tfidf_overview = None
        self.tfidf_keywords = None
        self.mlb_genre = None
        self.movie_data = None
        self.plot_embeddings = None
        self.title_matrix = None
        self.genre_matrix = None
        self.keyword_matrix = None
        self.overview_matrix = None

        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(self.script_dir, cache_dir_name)

        # 캐시 디렉토리 생성
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            print(f"캐시 디렉토리 생성: {self.cache_dir}")

        self.custom_stopwords = self.load_stopwords_from_files(stopword_files)

        genre_matrix_path = os.path.join(self.script_dir, "genre_similarity_matrix_content.json")
        with open(genre_matrix_path, "r", encoding="utf-8") as f:
            self.genre_similarity_matrix_content = json.load(f)
        
    def load_kobert_model(self):
        """KoBERT 모델 로드"""
        print("KoBERT 모델 로딩 중...")
        model_name = "monologg/kobert"
        self.kobert_model = AutoModel.from_pretrained(model_name)
        self.kobert_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.kobert_model.eval()
        print("KoBERT 모델 로드 완료")

    def enhanced_text_preprocessing(self, text):
        """텍스트 전처리 강화"""
        if not text or pd.isna(text):
            return ""
        
        text = str(text)
        
        # 특수 문자 및 숫자 처리
        text = re.sub(r'[^\w\s가-힣a-zA-Z0-9]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()

    def extract_nouns_from_text(self, text, okt):
        """텍스트에서 명사 추출"""
        if not text:
            return []
        
        # 명사 추출
        nouns = okt.nouns(text)
        
        # 단어 길이가 1 이하인 것 제거
        filtered_nouns = [noun for noun in nouns if len(noun) > 1]
        
        return filtered_nouns
    
    def load_stopwords_from_files(self, stopword_files=None):
        """스크립트 디렉토리를 기준으로 불용어 파일들을 로드합니다."""
        combined_stopwords = set()

        if stopword_files is None:
            stopwords_dir = os.path.join(self.script_dir, "stopwords")  # stopwords 디렉토리 경로

            try:
                stopword_files = [
                    os.path.join(stopwords_dir, fname)
                    for fname in os.listdir(stopwords_dir)
                    if fname.endswith(".txt")
                ]
            except FileNotFoundError:
                print(f"❌ stopwords 디렉토리를 찾을 수 없습니다: {stopwords_dir}")
                stopword_files = []

        for file_path in stopword_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        words = [line.strip() for line in f if line.strip()]
                        combined_stopwords.update(words)
                    print(f"✅ 불용어 파일 로드 완료: {file_path} ({len(words)}개)")
                except Exception as e:
                    print(f"❌ 불용어 파일 로드 실패: {file_path} - {e}")
            else:
                print(f"⚠️ 불용어 파일 없음: {file_path}")

        if not combined_stopwords:
            print("⚠️ 불용어 파일이 없거나 비어 있어 기본 불용어를 사용합니다.")
            return set(['하다', '있다', '없다', '되다', '이다', '것', '수', '점', '말', '안', '때', '등', '통해'])

        return combined_stopwords



    def enhanced_text_preprocessing_token(self, text):
        if pd.isna(text) or not text:
            return ""

        text = self.enhanced_text_preprocessing(text)
        stopwords_pos = ['Josa', 'Eomi', 'Suffix', 'Punctuation'] 
        
        tokens = self.okt.pos(text, norm=True, stem=True)
        
        filtered_tokens = []
        for word, pos in tokens:
            if pos not in stopwords_pos:
                # 함수 내부에서 로드한 custom_stopwords 사용
                if word not in self.custom_stopwords: 
                    filtered_tokens.append(word)
        
        return ' '.join(filtered_tokens)
    
    def extract_keywords(self, text):
        """텍스트에서 키워드 추출 (형태소 분석 + 불용어 제거 + 가중치 적용)"""
        if not text or pd.isna(text):
            return []
        
        # OKT 형태소 분석기 초기화 (클래스 변수로 관리하면 더 효율적)
        if not hasattr(self, 'okt'):
            self.okt = Okt()
        
        # 한국어 영화 관련 키워드 패턴 (가중치 포함)
        movie_keywords = {
            # 장르 관련 (가중치 2)
            'action': {
                'weight': 2,
                'keywords': [
                    '액션', '전투', '싸움', '전쟁', '격투', '추격', '폭발', '총격',
                    '무술', '카체이싱', '건파이트', '배틀', '액션스릴러', '느와르',
                    '스파이', '첩보', '잠입', '미션', '작전', '복수', '정의', '위장'
                ]
            },
            'romance': {
                'weight': 2,
                'keywords': [
                    '사랑', '로맨스', '연인', '결혼', '연애', '첫사랑', '이별', '만남',
                    '멜로', '러브스토리', '운명', '재회', '프로포즈', '웨딩', '데이트',
                    '썸', '고백', '짝사랑', '원거리', '국제연애', '나이차', '사내연애'
                ]
            },
            'comedy': {
                'weight': 2,
                'keywords': [
                    '코미디', '웃음', '유머', '재미', '개그', '유쾌', '농담',
                    '개그맨', '상황극', '슬랩스틱', '로맨틱코미디', '패밀리코미디',
                    '블랙코미디', '풍자', '해학', '익살', '코믹', '유머러스'
                ]
            },
            'horror': {
                'weight': 2,
                'keywords': [
                    '공포', '호러', '무서운', '귀신', '좀비', '괴물', '악령',
                    '사이코', '살인마', '연쇄살인', '초자연', '오컬트', '엑소시즘',
                    '저주', '원혼', '귀신', '유령', '무덤', '폐가', '심령', '위험구역'
                ]
            },
            'thriller': {
                'weight': 2,
                'keywords': [
                    '스릴러', '긴장', '추적', '수사', '범죄', '살인', '미스터리',
                    '서스펜스', '추리', '탐정', '형사', '검찰', '법정', '재판',
                    '납치', '협박', '음모', '배신', '사기', '해킹', '첩보'
                ]
            },
            'drama': {
                'weight': 2,
                'keywords': [
                    '인간', '감동', '눈물', '인생', '성장'
                ]
            },
            
            # 테마/소재 관련 (가중치 1.5)
            'theme': {
                'weight': 1.5,
                'keywords': [
                    '복수', '정의', '우정', '배신', '희생',
                    '성장', '자아실현', '꿈', '도전', '극복', '화해',
                    '갈등', '대립', '협력', '경쟁', '성공', '실패'
                ]
            },
            'emotion': {
                'weight': 1.5,
                'keywords': [
                    '감동', '눈물', '슬픔', '기쁨', '행복', '분노',
                    '절망', '희망', '사랑', '이별', '그리움', '향수',
                    '외로움', '고독', '우울', '스트레스', '끔찍한'
                ]
            },
            
            # 배경/직업 관련 (가중치 1)
            'profession': {
                'weight': 1,
                'keywords': [
                    '의사', '변호사', '교사', '경찰', '소방관', '군인',
                    '기자', '작가', '화가', '음악가', '요리사', '사업가',
                    '정치인', '연예인', '운동선수', '과학자', '엔지니어'
                ]
            },
            'background': {
                'weight': 1,
                'keywords': [
                    '현대', '도시', '서울', '학교', '대학', '회사', '직장',
                    '병원', '법원', '경찰서', '아파트', '카페'
                ]
            }
        }
        
        # 1. 텍스트 전처리
        preprocessed_text = self.enhanced_text_preprocessing(text)
        
        # 2. 형태소 분석으로 명사 추출
        nouns = self.extract_nouns_from_text(preprocessed_text, self.okt)
        
        # 3. 불용어 제거
        filtered_nouns = [noun for noun in nouns 
                        if noun not in self.custom_stopwords]
        
        # 4. 명사 빈도 계산
        noun_counts = Counter(filtered_nouns)
        
        # 5. 영화 키워드 매칭 및 가중치 적용
        keyword_scores = {}
        
        for category, category_info in movie_keywords.items():
            weight = category_info['weight']
            keywords = category_info['keywords']
            
            for keyword in keywords:
                # 키워드가 추출된 명사에 있는지 확인
                if keyword in filtered_nouns:
                    score = noun_counts[keyword] * weight
                    if keyword in keyword_scores:
                        keyword_scores[keyword] += score
                    else:
                        keyword_scores[keyword] = score
        
        # 6. 점수 기준으로 정렬하여 상위 키워드 반환
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 상위 20개 키워드만 반환 (키워드만, 점수 제외)
        top_keywords = [keyword for keyword, score in sorted_keywords[:20]]
        
        # 매칭되지 않은 중요 명사도 일부 포함 (빈도수 기준)
        remaining_nouns = [noun for noun in noun_counts.most_common(10) 
                        if noun[0] not in top_keywords]
        
        # 최종 키워드 결합
        final_keywords = top_keywords + [noun[0] for noun in remaining_nouns[:5]]
        
        return final_keywords[:25]  # 최대 25개 키워드 반환


    def get_kobert_embedding(self, text):
        """KoBERT를 사용하여 텍스트 임베딩 생성"""
        if not text or pd.isna(text):
            return np.zeros(768)  # KoBERT 임베딩 차원
        
        # 텍스트 길이 제한 (KoBERT 최대 입력 길이 고려)
        text = str(text)[:512]
        
        inputs = self.kobert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.kobert_model(**inputs)
            # [CLS] 토큰의 임베딩 사용
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().detach().cpu().numpy()
        
        return embedding
    
    def calculate_genre_similarity_advanced(self, target_genres, candidate_genres):
        """고급 장르 유사도 계산"""
        # 장르 간 유사도 매트릭스 (실제로는 더 정교하게 구성 가능)
        if not target_genres or not candidate_genres:
            return 0.0

        target_genres = [g.lower() for g in target_genres]
        candidate_genres = [g.lower() for g in candidate_genres]

        # 직접 매치 점수
        direct_match = len(set(target_genres) & set(candidate_genres)) / len(set(target_genres) | set(candidate_genres))
        
        # 간접 매치 점수
        indirect_score = 0
        for target_genre in target_genres:
            for candidate_genre in candidate_genres:
                if target_genre.lower() in self.genre_similarity_matrix_content:
                    if candidate_genre.lower() in self.genre_similarity_matrix_content[target_genre.lower()]:
                        indirect_score += self.genre_similarity_matrix_content[target_genre.lower()][candidate_genre.lower()]
        
        indirect_score = indirect_score / (len(target_genres) * len(candidate_genres)) if target_genres and candidate_genres else 0
        
        return 0.7 * direct_match + 0.3 * indirect_score
    
    def prepare_data(self, movies_df, force_recompute=False):
        """데이터 전처리 및 특성 추출"""
        print("데이터 전처리 중...")

         # 캐시 파일 경로 정의
        feature_objects_path = f"{self.cache_dir}/feature_objects.pkl"
        feature_matrices_path = f"{self.cache_dir}/feature_matrices.npz"
        movie_data_path = f"{self.cache_dir}/movie_data.pkl"
        plot_embeddings_path = f"{self.cache_dir}/plot_embeddings.npy" # KoBERT 임베딩도 여기에 포함

        # 캐시 존재 여부 확인 및 force_recompute에 따른 로드/재계산 결정
        if not force_recompute and \
           os.path.exists(feature_objects_path) and \
           os.path.exists(feature_matrices_path) and \
           os.path.exists(movie_data_path) and \
           os.path.exists(plot_embeddings_path): # KoBERT 임베딩 파일도 확인
            
            print("✅ 저장된 특성 데이터를 불러옵니다...")
            try:
                # 1. Feature objects (Vectorizer, SVD, MLB) 로드
                with open(feature_objects_path, 'rb') as f:
                    feature_objects = pickle.load(f)
                    self.tfidf_title = feature_objects['tfidf_title']
                    self.tfidf_overview = feature_objects['tfidf_overview']
                    self.tfidf_keywords = feature_objects['tfidf_keywords']
                    self.svd = feature_objects['svd']
                    self.mlb_genre = feature_objects['mlb_genre']

                # 2. Feature matrices (NumPy arrays) 로드
                loaded_matrices = np.load(feature_matrices_path, allow_pickle=True)
                self.title_matrix = loaded_matrices['title_matrix']
                self.overview_matrix = loaded_matrices['overview_matrix']
                self.keyword_matrix = loaded_matrices['keyword_matrix']
                self.genre_matrix = loaded_matrices['genre_matrix']

                # 3. KoBERT 임베딩 로드 (별도 파일로 관리)
                self.plot_embeddings = np.load(plot_embeddings_path)

                # 4. 영화 데이터 로드
                self.movie_data = pd.read_pickle(movie_data_path)

                print("✅ 모든 특성 데이터 로드 완료.")
                return

            except Exception as e:
                print(f"❌ 저장된 특성 데이터 로드 중 오류 발생: {e}. 데이터를 새로 계산합니다.")
                # 오류 발생 시 다시 계산하도록 플래그 설정
                force_recompute = True # 에러 발생 시 재계산을 유도

        # 캐시가 없거나, 강제로 재계산해야 하는 경우
        print("🛠 데이터를 새로 전처리하고 특성을 추출합니다...")
        
        # 필수 컬럼 확인
        required_columns = ['title', 'overview', 'genres']
        for col in required_columns:
            if col not in movies_df.columns:
                raise ValueError(f"필수 컬럼 '{col}'이 데이터에 없습니다.")
        
        # 결측값 처리
        movies_df['title'] = movies_df['title'].fillna('')
        movies_df['overview'] = movies_df['overview'].fillna('')
        movies_df['genres'] = movies_df['genres'].fillna('')
        
        # 키워드 추출
        print("키워드 추출 중...")
        movies_df['keywords'] = movies_df['overview'].apply(self.extract_keywords)
        movies_df['keywords_text'] = movies_df['keywords'].apply(lambda x: ' '.join(x) if x else '')
        
        movies_df['title_text'] = movies_df['title'].apply(self.enhanced_text_preprocessing_token)
        movies_df['overview_text'] = movies_df['overview'].apply(self.enhanced_text_preprocessing_token)

        self.movie_data = movies_df.copy()
        
        # 1. 제목 TF-IDF 벡터화
        print("제목 TF-IDF 벡터화 중...")
        self.tfidf_title = TfidfVectorizer(
            max_features=5000,
            stop_words=None,
            ngram_range=(1, 2),
            min_df=1
        )
        self.title_matrix = self.tfidf_title.fit_transform(movies_df['title_text'])

        # 2. 줄거리 TF-IDF 벡터화 (형태소 분리 적용)    
        self.tfidf_overview = TfidfVectorizer(
            max_features=10000, 
            stop_words=None, 
            ngram_range=(1, 2), 
            min_df=1 
        )
        overview_tfidf_matrix = self.tfidf_overview.fit_transform(movies_df['overview_text'])
        print("줄거리 TF-IDF 벡터화 완료.")

        # LSA (Truncated SVD)
        self.svd = TruncatedSVD(n_components=100, random_state=42)
        self.overview_matrix = self.svd.fit_transform(overview_tfidf_matrix)
        
        # 3. 키워드 TF-IDF 벡터화
        print("키워드 TF-IDF 벡터화 중...")
        self.tfidf_keywords = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 1),
            min_df=3,  # 최소 2개 영화에서 나타나는 키워드만 사용
            max_df=0.4,  # 40% 이상 영화에 나타나는 키워드는 제외
        )
        self.keyword_matrix = self.tfidf_keywords.fit_transform(movies_df['keywords_text'])
        
        # 4. 장르 처리
        print("장르 처리 중...")
        genre_lists = []
        for genres in movies_df['genres']:
            if isinstance(genres, str) and genres:
                genre_list = [g.strip() for g in genres.split(',')]
                genre_lists.append(genre_list)
            elif isinstance(genres, list):
                genre_lists.append(genres)
            else:
                genre_lists.append([])
        
        self.mlb_genre = MultiLabelBinarizer()
        self.genre_matrix = self.mlb_genre.fit_transform(genre_lists)

        # 5. KoBERT 임베딩
        print("🛠 줄거리 임베딩을 새로 생성합니다.")
        if self.kobert_model is None:
            self.load_kobert_model()
        
        plot_embeddings = []
        for i, overview in enumerate(movies_df['overview_text']):
            if i % 50 == 0:
                print(f"임베딩 진행률: {i}/{len(movies_df)}")
            embedding = self.get_kobert_embedding(overview)
            plot_embeddings.append(embedding)

        self.plot_embeddings = np.array(plot_embeddings)
        print("줄거리 임베딩 생성 완료.")

        self._save_all_features()
        print("데이터 전처리 완료")

    def _save_all_features(self):
        """모든 특성을 한 번에 저장"""
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        print("💾 모든 특성 데이터 저장 중...")
        
        # 1. TF-IDF 벡터화 객체들 저장
        feature_objects = {
            'tfidf_title': self.tfidf_title,
            'tfidf_overview': self.tfidf_overview,
            'tfidf_keywords': self.tfidf_keywords,
            'svd': self.svd,
            'mlb_genre': self.mlb_genre
        }
        
        # Ensure file exists before attempting to open, or handle errors
        try:
            with open(f"{self.cache_dir}/feature_objects.pkl", 'wb') as f:
                pickle.dump(feature_objects, f)
        except Exception as e:
            print(f"Error saving feature objects: {e}")
            # Optionally, re-raise or handle more gracefully

        # 2. 특성 행렬들 저장
        # Scipy sparse matrix (CSR) toarray() to save as dense numpy array
        # Check if matrices are sparse before converting
        title_matrix_array = self.title_matrix.toarray() if hasattr(self.title_matrix, 'toarray') else self.title_matrix
        keyword_matrix_array = self.keyword_matrix.toarray() if hasattr(self.keyword_matrix, 'toarray') else self.keyword_matrix
        overview_matrix_array = self.overview_matrix if isinstance(self.overview_matrix, np.ndarray) else self.overview_matrix.toarray()


        np.savez_compressed(f"{self.cache_dir}/feature_matrices.npz",
                            title_matrix=title_matrix_array,
                            overview_matrix=overview_matrix_array,
                            keyword_matrix=keyword_matrix_array,
                            genre_matrix=self.genre_matrix)
        
        # KoBERT 임베딩은 별도 파일로 저장
        np.save(f"{self.cache_dir}/plot_embeddings.npy", self.plot_embeddings)

        # 3. 영화 데이터 저장 (전처리된 컬럼 포함)
        self.movie_data.to_pickle(f"{self.cache_dir}/movie_data.pkl")
        
        print(f"✅ 모든 데이터가 {self.cache_dir}에 저장되었습니다.")
    
    def calculate_similarity_comprehensive(self, movie_idx, weights):
        """포괄적인 유사도 계산"""
        
        # 1. KoBERT 임베딩 기반 유사도 (의미적 유사도)
        target_plot_embedding = self.plot_embeddings[movie_idx].reshape(1, -1)
        kobert_similarities = cosine_similarity(target_plot_embedding, self.plot_embeddings)[0]
        kobert_similarities[kobert_similarities < 0] = 0
        
        # 2. 줄거리 TF-IDF 기반 유사도 (어휘적 유사도)
        target_overview_vector = self.overview_matrix[movie_idx]
        overview_similarities = cosine_similarity(target_overview_vector.reshape(1, -1), self.overview_matrix)[0]
        overview_similarities[overview_similarities < 0] = 0
        
        # 3. 제목 유사도
        target_title_vector = self.title_matrix[movie_idx]
        title_similarities = cosine_similarity(target_title_vector.reshape(1, -1), self.title_matrix)[0]
        
        # 4. 키워드 유사도
        target_keyword_vector = self.keyword_matrix[movie_idx]
        keyword_similarities = cosine_similarity(target_keyword_vector.reshape(1, -1), self.keyword_matrix)[0]
        
        # 5. 장르 유사도
        target_genres = self.movie_data.iloc[movie_idx]['genres'].split(',') if self.movie_data.iloc[movie_idx]['genres'] else []
        target_genres = [g.strip() for g in target_genres]
        
        genre_similarities = []
        for i in range(len(self.movie_data)):
            candidate_genres = self.movie_data.iloc[i]['genres'].split(',') if self.movie_data.iloc[i]['genres'] else []
            candidate_genres = [g.strip() for g in candidate_genres]
            
            similarity = self.calculate_genre_similarity_advanced(target_genres, candidate_genres)
            genre_similarities.append(similarity)
        
        genre_similarities = np.array(genre_similarities)
        
        # 6. 가중 평균으로 최종 유사도 계산
        final_similarities = (
            weights['kobert'] * kobert_similarities +
            weights['overview_tfidf'] * overview_similarities +
            weights['title'] * title_similarities +
            weights['keywords'] * keyword_similarities +
            weights['genre'] * genre_similarities
        )
        
        return (final_similarities, kobert_similarities, overview_similarities, 
                title_similarities, keyword_similarities, genre_similarities)
    
    def adaptive_weights(self, movie_idx):
        """영화 특성에 따른 적응적 가중치 계산"""
        movie_info = self.movie_data.iloc[movie_idx]
        
        # 1. 기본 가중치 설정 (총합 100%)
        weights = {
            'kobert': 0.20,
            'overview_tfidf': 0.35,
            'title': 0.10,  # 제목 유사도 문제 해결 전까지는 낮게 유지
            'keywords': 0.20, # 키워드 중요성을 높임
            'genre': 0.25   # 장르 중요성을 높임
        }

        # 모든 가중치 조정은 상대적으로 이루어지므로, 총합이 1이 되도록 조정하는 것이 중요합니다.
        # 여기서는 편의상 절대값으로 더하고 빼지만, 실제로는 비율로 조정하거나, 마지막에 정규화해야 합니다.


        # 2. 줄거리 길이에 따른 조정
        overview_length = len(movie_info['overview']) if movie_info['overview'] else 0

        if overview_length > 300:  # 긴 줄거리: 줄거리 관련 가중치 강화
            weights['kobert'] += 0.03
            weights['overview_tfidf'] += 0.05
            weights['genre'] -= 0.04 # 상대적으로 장르 중요도 감소
            weights['keywords'] -= 0.04
        elif overview_length < 50:  # 짧은 줄거리: 제목/장르/키워드 강화, 줄거리 관련 약화
            weights['title'] += 0.05 # 제목 유사도 개선 시 효과적
            weights['genre'] += 0.05
            weights['keywords'] += 0.05
            weights['kobert'] -= 0.07 # KoBERT 낮춤
            weights['overview_tfidf'] -= 0.08 # TF-IDF 낮춤

        # 3. 장르 수에 따른 조정 (if-elif-else 구조 사용)
        genre_count = len(movie_info['genres'].split(',')) if movie_info['genres'] else 0

        if genre_count == 1:  # 단일 장르: 해당 장르의 특성을 강하게 반영
            weights['genre'] += 0.10
            weights['overview_tfidf'] -= 0.05
            weights['keywords'] -= 0.05
            weights['kobert'] -= 0.02
        elif genre_count > 3:  # 많은 장르: 특정 장르에 덜 치우치도록
            weights['genre'] -= 0.03
            weights['kobert'] += 0.03 # KoBERT 영향력을 다시 높여 전체 줄거리 문맥 반영
        # else: (2개의 장르인 경우 기본 가중치 유지)

        # 4. 특정 장르에 대한 조정 (가장 구체적인 조건부터 배치)
        genres = movie_info['genres'].lower() if movie_info['genres'] else ''

        # 스릴러와 드라마 조합 (기생충, 하녀 등)
        if '드라마' in genres and '스릴러' in genres and '코미디' in genres: # 기생충 같은 경우
            weights['genre'] += 0.15 # 장르 매우 중요
            weights['keywords'] += 0.05
            weights['kobert'] -= 0.05 # KoBERT 비중 감소
        elif '드라마' in genres and '스릴러' in genres: # 일반적인 드라마/스릴러
            weights['genre'] += 0.10
            weights['keywords'] += 0.05
            weights['overview_tfidf'] += 0.02 # 줄거리 핵심 단어 중요도도 높임
        elif 'sf' in genres or '공포' in genres: # '괴물' 같은 경우
            weights['genre'] += 0.15 # 장르 매우 중요
            weights['keywords'] += 0.10 # SF/공포는 키워드가 중요 (괴물, 좀비 등)
            weights['overview_tfidf'] += 0.05 # 줄거리 내 핵심 단어 중요
            weights['kobert'] -= 0.10 # KoBERT 비중 대폭 감소
        elif '액션' in genres:
            weights['keywords'] += 0.05
            weights['overview_tfidf'] += 0.05 # 액션은 줄거리 내 동작 묘사가 중요
            weights['title'] -= 0.05 # 제목보다는 내용이 중요
        elif '로맨스' in genres:
            weights['kobert'] += 0.03 # 감성적인 줄거리 유사성 중요
            weights['overview_tfidf'] += 0.03
            weights['keywords'] -= 0.03
            weights['genre'] -= 0.03 # 로맨스는 장르가 넓어질 수 있으므로
        elif '코미디' in genres:
            weights['kobert'] += 0.02 # 코미디도 줄거리의 뉘앙스가 중요
            weights['genre'] += 0.05 # 코미디 장르의 특색
            weights['keywords'] -= 0.02

        # 음수 가중치 방지 및 최소값 설정 (선택 사항)
        for key in weights:
            if weights[key] < 0.01: # 최소 가중치 설정 (예: 1%)
                weights[key] = 0.01

        # 가중치 정규화
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights

    def recommend_movies(self, movie_idx=None, top_n=10, weights=None, use_adaptive_weights=True, content_id=None, content_type=None):
       
        if movie_idx is None:
            # 복합키를 통한 인덱스 탐색
            if content_id is not None and content_type is not None:
                matches = self.movie_data[
                    (self.movie_data['id'] == content_id) &
                    (self.movie_data['type'] == content_type)
                ]
                if matches.empty:
                    print(f"id '{content_id}'와 type '{content_type}'에 해당하는 영화를 찾을 수 없습니다.")
                    return None
                movie_idx = matches.index[0]
                actual_title = matches.iloc[0]['title']

            else:
                raise ValueError("movie_idx 또는 contentId + type 또는 movie_title 중 하나는 제공되어야 합니다.")
        else:
            actual_title = self.movie_data.iloc[movie_idx]['title']

        # 가중치 설정
        if use_adaptive_weights:
            weights = self.adaptive_weights(movie_idx)
            print(f"적응적 가중치 사용: {weights}")
        elif weights is None:
            weights = {
                'kobert': 0.15,
                'overview_tfidf': 0.35,
                'title': 0.10,
                'keywords': 0.25,
                'genre': 0.15
            }

        print(f"\n기준 영화: {actual_title}")
        print(f"장르: {self.movie_data.iloc[movie_idx]['genres']}")
        print(f"줄거리: {self.movie_data.iloc[movie_idx]['overview'][:400]}...")
        print(f"   키워드: {self.movie_data.iloc[movie_idx]['keywords']}")

        # 유사도 계산
        similarities = self.calculate_similarity_comprehensive(movie_idx, weights)
        final_similarities = similarities[0]

        # 자기 자신 제외 상위 N개
        similar_indices = np.argsort(final_similarities)[::-1]
        similar_indices = [idx for idx in similar_indices if idx != movie_idx][:top_n]

        # 추천 결과
        recommendations = []
        for idx in similar_indices:
            movie_info = {
                'content_id': self.movie_data.iloc[idx].get('id'),
                'content_type': self.movie_data.iloc[idx].get('type'),
                'title': self.movie_data.iloc[idx]['title'],
                'genres': self.movie_data.iloc[idx]['genres'],
                'overview': self.movie_data.iloc[idx]['overview'],
                'keywords': self.movie_data.iloc[idx]['keywords'],
                'total_similarity': final_similarities[idx],
                'poster_path': self.movie_data.iloc[idx]['poster_path'],
            }
            recommendations.append(movie_info)

        return recommendations

    
    def display_recommendations(self, recommendations):
        """추천 결과 출력"""
        if not recommendations:
            print("추천할 영화가 없습니다.")
            return
        
        print(f"\n=== 추천 영화 TOP {len(recommendations)} ===")
        
        for i, movie in enumerate(recommendations, 1):
            print(f"\n{i}. {movie['title']}")
            print(f"content_id: {movie['content_id']}, content_type: {movie['content_type']}")
            print(f"   총 유사도: {movie['total_similarity']:.3f}")
            print(f"   콘텐츠 포스터: {movie['poster_path']}")

if __name__ == "__main__":
    # 데이터셋 로드 (실제 경로에 맞게 수정 필요)
    movies_df = pd.read_csv('content_data.csv', encoding='utf-8-sig')
    print(f"✅ 'content_data.csv' 파일 로드 성공. 총 {len(movies_df)}개 데이터.")
    
    # 개선된 추천 시스템 초기화
    recommender = ImprovedMovieRecommendationSystem(cache_dir_name="./cached_features")
    
    # 데이터 전처리
    recommender.prepare_data(movies_df)
    
    # 적응적 가중치로 추천 실행
    recommendations1 = recommender.recommend_movies(
        content_id= 5707,
        content_type= "tv",
        top_n=8,
        use_adaptive_weights=True
    )
    recommender.display_recommendations(recommendations1)