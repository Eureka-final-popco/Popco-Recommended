import pandas as pd
from typing import Optional, Any, Dict, Tuple, List
import logging

logger = logging.getLogger(__name__)

class PBRAppState:
    def __init__(self):
        # 데이터프레임
        self.contents_df: Optional[pd.DataFrame] = None
        self.reactions_df: Optional[pd.DataFrame] = None
        self.reviews_df: Optional[pd.DataFrame] = None
        self.all_user_personas_df: Optional[pd.DataFrame] = None
        self.persona_df: Optional[pd.DataFrame] = None
        self.persona_genres_df: Optional[pd.DataFrame] = None
        self.genres_df: Optional[pd.DataFrame] = None
        self.persona_options_df: Optional[pd.DataFrame] = None
        self.options_df: Optional[pd.DataFrame] = None
        self.user_qa_answers_df: Optional[pd.DataFrame] = None

        # 매핑
        self.user_id_to_idx_map: Optional[Dict[int, int]] = None
        self.user_idx_to_id_map: Optional[Dict[int, int]] = None
        self.content_id_to_idx_map: Optional[Dict[Tuple[int, str], int]] = None
        self.idx_to_content_id_map: Optional[Dict[int, Tuple[int, str]]] = None

        self.persona_name_to_id_map: Optional[Dict[str, int]] = None
        self.persona_id_to_name_map: Optional[Dict[int, str]] = None
        self.genre_id_to_name_map: Optional[Dict[int, str]] = None
        self.persona_details_map: Optional[Dict[int, Dict[str, Any]]] = None
        self.all_user_ids: List[int] = []

        # 유사도 매트릭스
        self.user_item_matrix: Optional[Any] = None
        self.user_similarity_df: Optional[pd.DataFrame] = None
        self.user_similarity_matrix: Optional[Any] = None
        self.content_similarity_matrix: Optional[Any] = None

        self.persona_similarity_df: Optional[pd.DataFrame] = None

        # 임베딩 관련
        self.user_persona_interaction_matrix = None
        self.content_genre_embedding_model = None
        self.content_embedding_model = None
        self.content_embeddings = None
        
        # 캐시 및 기타
        self.persona_keyword_map: Dict[int, List[str]] = {}
        self.recommendation_cache: Dict[int, Dict[str, Any]] = {}


pbr_app_state = PBRAppState()
logger.info("[PBRAppState] pbr_app_state 인스턴스가 생성되었습니다.")