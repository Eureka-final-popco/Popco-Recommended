import os
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Path
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Tuple
from scipy.sparse import csr_matrix 
from .data_loader import load_initial_data, load_user_personas, save_user_personas
from .persona_generator import (
    generate_unified_user_feedback,
    get_hybrid_persona,
    generate_initial_user_personas,
    calculate_user_content_matrix_sparse,
    calculate_user_similarity,
    recommend_contents_cf,
    GLOBAL_USER_ID_TO_IDX,
    GLOBAL_USER_IDX_TO_ID,
    GLOBAL_CONTENT_ID_TO_IDX,
    GLOBAL_CONTENT_IDX_TO_ID,
    _get_user_idx_from_id, 
    _get_user_id_from_idx,
    _get_content_idx_from_id, 
    _get_content_id_from_idx 
)
from .config import default_personas, content_genre_keywords_mapping, MIN_FEEDBACK_FOR_PERSONA_DETERMINATION


class RecommendationItem(BaseModel):
    contentId: int
    title: str
    genres: List[str]
    predicted_rating: float
    persona_genre_match: bool

class InitialFeedbackRequest(BaseModel):
    user_id: int
    content_ids: List[int]
    reaction_type: str
    initial_answers: Dict[str, str]

class FeedbackRequest(BaseModel):
    user_id: int
    content_id: int
    reaction_type: str
    score: Optional[float] = None

class RecommendationResponse(BaseModel):
    message: str
    recommendations: List[RecommendationItem]
    main_persona: str
    sub_persona: str
    all_personas_scores: Dict[str, float]

class PersonaInfoResponse(BaseModel):
    user_id: int
    main_persona: str
    sub_persona: str
    all_personas_scores: Dict[str, float]


app_state: Dict[str, Any] = {
    "contents_df_persona": pd.DataFrame(),
    "current_virtual_reactions_df": pd.DataFrame(),
    "current_virtual_reviews_df": pd.DataFrame(),
    "current_user_content_matrix_cf": None,
    "current_user_similarity_df_cf": pd.DataFrame(),
    "current_all_user_personas_df": pd.DataFrame()
}

# FastAPI Lifespan 함수
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("추천 시스템 데이터를 초기화하는 중입니다...")

    current_script_path = os.path.abspath(__file__)
    current_script_dir = os.path.dirname(current_script_path)
    app_dir = os.path.dirname(current_script_dir)
    data_processing_path = os.path.join(app_dir, 'data_processing')
    user_persona_csv_absolute_path = os.path.join(data_processing_path, 'user_personas.csv')

    print(f"데이터를 '{data_processing_path}'에서 로드 중입니다...")

    contents_df_persona, content_similarity_df, reviews_df, content_reactions_df = \
        load_initial_data(data_path=data_processing_path)

    all_unique_user_ids = pd.concat([content_reactions_df['userId'], reviews_df['userId']]).unique()
    all_unique_content_ids = pd.concat([content_reactions_df['contentId'], reviews_df['contentId'], contents_df_persona['contentId']]).unique()

    for user_id in all_unique_user_ids:
        if user_id not in GLOBAL_USER_ID_TO_IDX:
            idx = len(GLOBAL_USER_ID_TO_IDX)
            GLOBAL_USER_ID_TO_IDX[user_id] = idx
            GLOBAL_USER_IDX_TO_ID[idx] = user_id

    for content_id in all_unique_content_ids:
        if content_id not in GLOBAL_CONTENT_ID_TO_IDX:
            idx = len(GLOBAL_CONTENT_ID_TO_IDX)
            GLOBAL_CONTENT_ID_TO_IDX[content_id] = idx
            GLOBAL_CONTENT_IDX_TO_ID[idx] = content_id

    print(f"매핑된 사용자 수: {len(GLOBAL_USER_ID_TO_IDX)}, 매핑된 콘텐츠 수: {len(GLOBAL_CONTENT_ID_TO_IDX)}")

    content_reactions_df['userId_orig'] = content_reactions_df['userId']
    content_reactions_df['userId'] = content_reactions_df['userId'].map(GLOBAL_USER_ID_TO_IDX)
    content_reactions_df['contentId_orig'] = content_reactions_df['contentId']
    content_reactions_df['contentId'] = content_reactions_df['contentId'].map(GLOBAL_CONTENT_ID_TO_IDX)

    reviews_df['userId_orig'] = reviews_df['userId']
    reviews_df['userId'] = reviews_df['userId'].map(GLOBAL_USER_ID_TO_IDX)
    reviews_df['contentId_orig'] = reviews_df['contentId']
    reviews_df['contentId'] = reviews_df['contentId'].map(GLOBAL_CONTENT_ID_TO_IDX)

    contents_df_persona['contentId_orig'] = contents_df_persona['contentId']
    contents_df_persona['contentId'] = contents_df_persona['contentId'].map(GLOBAL_CONTENT_ID_TO_IDX)

    contents_df_persona['genres'] = contents_df_persona['genres'].apply(
        lambda x: x.split('|') if isinstance(x, str) else x
    )

    app_state["contents_df_persona"] = contents_df_persona 
    app_state["current_virtual_reactions_df"] = content_reactions_df 
    app_state["current_virtual_reviews_df"] = reviews_df 

    app_state["current_user_content_matrix_cf"] = calculate_user_content_matrix_sparse(
        app_state["current_virtual_reactions_df"],
        app_state["current_virtual_reviews_df"],
        app_state["contents_df_persona"] 
    )

    if app_state["current_user_content_matrix_cf"].shape[0] >= 2 and app_state["current_user_content_matrix_cf"].shape[1] >= 1:
        app_state["current_user_similarity_df_cf"] = calculate_user_similarity(app_state["current_user_content_matrix_cf"])
    else:
        app_state["current_user_similarity_df_cf"] = pd.DataFrame()

    all_user_personas_df_loaded = load_user_personas(user_data_path=user_persona_csv_absolute_path)


    if all_user_personas_df_loaded.empty:
        print("[정보] user_personas.csv 파일이 비어있거나 존재하지 않아, 초기 사용자 페르소나를 생성합니다.")
        mapped_user_indices_for_persona_calc = list(GLOBAL_USER_IDX_TO_ID.keys()) 
        all_user_personas_df_generated = generate_initial_user_personas(
            mapped_user_indices_for_persona_calc,
            app_state["current_virtual_reactions_df"], 
            app_state["current_virtual_reviews_df"],   
            app_state["contents_df_persona"],          
            default_personas,
            content_genre_keywords_mapping,
            initial_answers=None
        )
        app_state["current_all_user_personas_df"] = all_user_personas_df_generated
        save_user_personas(app_state["current_all_user_personas_df"], user_persona_csv_absolute_path)
    else:
        print("[정보] 기존 user_personas.csv 파일에서 사용자 페르소나를 로드했습니다.")
        app_state["current_all_user_personas_df"] = all_user_personas_df_loaded

    print("추천 시스템 데이터 초기화 및 준비 완료.")
    yield
    print("추천 시스템 데이터 종료 및 상태 저장 중...")
    save_user_personas(app_state["current_all_user_personas_df"], user_persona_csv_absolute_path)
    print("추천 시스템 데이터가 저장되었습니다.")


app = FastAPI(lifespan=lifespan)

# 페르소나 정보 추출 로직
def get_user_persona_details(user_id: int) -> Dict[str, Any]:
    unified_feedback_for_user = generate_unified_user_feedback(
        user_id, 
        app_state["current_virtual_reactions_df"],
        app_state["current_virtual_reviews_df"],
        app_state["contents_df_persona"]
    )
    main_persona_name, sub_persona_name, all_personas_scores = get_hybrid_persona(
        user_id, 
        app_state["current_all_user_personas_df"],
        unified_feedback_for_user,
        app_state["contents_df_persona"], 
        default_personas,
        content_genre_keywords_mapping
    )

    return {
        "main_persona": main_persona_name,
        "sub_persona": sub_persona_name,
        "all_personas_scores": all_personas_scores
    }


@app.post("/onboard", response_model=RecommendationResponse, summary="새 사용자 초기 설문 및 선호 콘텐츠 등록")
async def onboard_user(request: InitialFeedbackRequest):
    user_id = request.user_id 
    content_ids = request.content_ids 
    reaction_type = request.reaction_type
    initial_answers = request.initial_answers

    if reaction_type != "좋아요":
        raise HTTPException(status_code=400, detail="온보딩 시에는 '좋아요' 반응만 허용됩니다.")
    if not initial_answers:
        raise HTTPException(status_code=400, detail="온보딩 시에는 초기 설문 답변이 필수입니다.")
    if not content_ids or len(content_ids) < 1:
        raise HTTPException(status_code=400, detail="온보딩 시에는 선호하는 콘텐츠 ID가 최소 1개 필요합니다.")

    print(f"새 사용자 {user_id} 온보딩 처리 중: 콘텐츠들={content_ids}, 반응={reaction_type}, 초기 답변 제공됨.")

    if user_id not in GLOBAL_USER_ID_TO_IDX:
        idx = len(GLOBAL_USER_ID_TO_IDX)
        GLOBAL_USER_ID_TO_IDX[user_id] = idx
        GLOBAL_USER_IDX_TO_ID[idx] = user_id

    for cid in content_ids:
        if cid not in GLOBAL_CONTENT_ID_TO_IDX:
            idx = len(GLOBAL_CONTENT_ID_TO_IDX)
            GLOBAL_CONTENT_ID_TO_IDX[cid] = idx
            GLOBAL_CONTENT_IDX_TO_ID[idx] = cid

    new_reactions_data = [
        {'userId_orig': user_id, 'userId': _get_user_idx_from_id(user_id),
         'contentId_orig': cid, 'contentId': _get_content_idx_from_id(cid),
         'reaction': reaction_type}
        for cid in content_ids
    ]
    new_reactions_df = pd.DataFrame(new_reactions_data)

    app_state["current_virtual_reactions_df"] = pd.concat([app_state["current_virtual_reactions_df"], new_reactions_df], ignore_index=True)

    # 1. 사용자-콘텐츠 행렬 재계산
    app_state["current_user_content_matrix_cf"] = calculate_user_content_matrix_sparse(
        app_state["current_virtual_reactions_df"],
        app_state["current_virtual_reviews_df"],
        app_state["contents_df_persona"]
    )

    # 2. 사용자 유사도 재계산
    if app_state["current_user_content_matrix_cf"].shape[0] >= 2 and app_state["current_user_content_matrix_cf"].shape[1] >= 1:
        app_state["current_user_similarity_df_cf"] = calculate_user_similarity(app_state["current_user_content_matrix_cf"])
    else:
        app_state["current_user_similarity_df_cf"] = pd.DataFrame() 

    # 3. 사용자 페르소나 재계산
    updated_user_persona_df = generate_initial_user_personas(
        [_get_user_idx_from_id(user_id)], 
        app_state["current_virtual_reactions_df"],
        app_state["current_virtual_reviews_df"],
        app_state["contents_df_persona"],
        default_personas,
        content_genre_keywords_mapping,
        initial_answers=initial_answers
    )

    if not updated_user_persona_df.empty:
        app_state["current_all_user_personas_df"] = app_state["current_all_user_personas_df"][
            app_state["current_all_user_personas_df"]['user_id'] != user_id
        ]
        app_state["current_all_user_personas_df"] = pd.concat(
            [app_state["current_all_user_personas_df"], updated_user_persona_df], ignore_index=True
        )

    persona_details = get_user_persona_details(user_id)

    final_recommendations_list_with_details = recommend_contents_cf(
        target_user_id=user_id,
        user_content_matrix=app_state["current_user_content_matrix_cf"],
        user_similarity_df=app_state["current_user_similarity_df_cf"],
        contents_df_persona=app_state["contents_df_persona"],
        all_user_personas_df=app_state["current_all_user_personas_df"],
        reactions_df=app_state["current_virtual_reactions_df"],
        reviews_df=app_state["current_virtual_reviews_df"],
        num_recommendations=10
    )

    return RecommendationResponse(
        message="온보딩이 완료되었고 초기 추천 목록이 제공되었습니다.",
        recommendations=final_recommendations_list_with_details,
        main_persona=persona_details["main_persona"],
        sub_persona=persona_details["sub_persona"],
        all_personas_scores=persona_details["all_personas_scores"]
    )


@app.post("/feedback", response_model=RecommendationResponse, summary="기존 사용자 피드백 업데이트 및 추천 목록 반환")
async def update_user_feedback(request: FeedbackRequest):
    user_id = request.user_id
    content_id = request.content_id
    reaction_type = request.reaction_type
    score = request.score

    if user_id not in GLOBAL_USER_ID_TO_IDX:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found. Please onboard first or ensure user data is loaded.")

    if content_id not in GLOBAL_CONTENT_ID_TO_IDX:
        idx = len(GLOBAL_CONTENT_ID_TO_IDX)
        GLOBAL_CONTENT_ID_TO_IDX[content_id] = idx
        GLOBAL_CONTENT_IDX_TO_ID[idx] = content_id

    print(f"사용자 {user_id}의 피드백을 업데이트 중: 콘텐츠={content_id}, 반응={reaction_type}, 점수={score}")

    mapped_user_id = _get_user_idx_from_id(user_id)
    mapped_content_id = _get_content_idx_from_id(content_id)

    if reaction_type == "평점":
        if score is None:
            raise HTTPException(status_code=400, detail="'평점' 반응 시에는 score 필드가 필수입니다.")
        new_review = pd.DataFrame([{'userId_orig': user_id, 'userId': mapped_user_id,
                                    'contentId_orig': content_id, 'contentId': mapped_content_id,
                                    'score': score}])
        app_state["current_virtual_reviews_df"] = pd.concat([app_state["current_virtual_reviews_df"], new_review], ignore_index=True)
    else:
        new_reaction = pd.DataFrame([{'userId_orig': user_id, 'userId': mapped_user_id,
                                      'contentId_orig': content_id, 'contentId': mapped_content_id,
                                      'reaction': reaction_type}])
        app_state["current_virtual_reactions_df"] = pd.concat([app_state["current_virtual_reactions_df"], new_reaction], ignore_index=True)

    # 1. 사용자-콘텐츠 행렬 재계산
    app_state["current_user_content_matrix_cf"] = calculate_user_content_matrix_sparse(
        app_state["current_virtual_reactions_df"],
        app_state["current_virtual_reviews_df"],
        app_state["contents_df_persona"]
    )

    # 2. 사용자 유사도 재계산
    if app_state["current_user_content_matrix_cf"].shape[0] >= 2 and app_state["current_user_content_matrix_cf"].shape[1] >= 1:
        app_state["current_user_similarity_df_cf"] = calculate_user_similarity(app_state["current_user_content_matrix_cf"])
    else:
        app_state["current_user_similarity_df_cf"] = pd.DataFrame() 

    # 3. 사용자 페르소나 재계산
    updated_user_persona_df = generate_initial_user_personas(
        [mapped_user_id],
        app_state["current_virtual_reactions_df"],
        app_state["current_virtual_reviews_df"],
        app_state["contents_df_persona"],
        default_personas,
        content_genre_keywords_mapping,
        initial_answers=None
    )

    if not updated_user_persona_df.empty:
        app_state["current_all_user_personas_df"] = app_state["current_all_user_personas_df"][
            app_state["current_all_user_personas_df"]['user_id'] != user_id
        ]
        app_state["current_all_user_personas_df"] = pd.concat(
            [app_state["current_all_user_personas_df"], updated_user_persona_df], ignore_index=True
        )

    persona_details = get_user_persona_details(user_id)

    final_recommendations_list_with_details = recommend_contents_cf(
        target_user_id=user_id,
        user_content_matrix=app_state["current_user_content_matrix_cf"],
        user_similarity_df=app_state["current_user_similarity_df_cf"],
        contents_df_persona=app_state["contents_df_persona"],
        all_user_personas_df=app_state["current_all_user_personas_df"],
        reactions_df=app_state["current_virtual_reactions_df"],
        reviews_df=app_state["current_virtual_reviews_df"],
        num_recommendations=10
    )

    print(f"사용자 {user_id}의 피드백이 성공적으로 업데이트되었고, 최신 추천이 반환됩니다.")
    return RecommendationResponse(
        message=f"사용자 {user_id}의 피드백이 성공적으로 업데이트되었고, 최신 추천 목록이 제공되었습니다.",
        recommendations=final_recommendations_list_with_details,
        main_persona=persona_details["main_persona"],
        sub_persona=persona_details["sub_persona"],
        all_personas_scores=persona_details["all_personas_scores"]
    )

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse, summary="사용자 페르소나 및 추천 목록 조회")
async def get_user_recommendations(user_id: int = Path(..., description="추천을 조회할 사용자 ID")):
    if user_id not in GLOBAL_USER_ID_TO_IDX:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found or no persona data. Please onboard first.")

    persona_details = get_user_persona_details(user_id)

    final_recommendations_list_with_details = recommend_contents_cf(
        target_user_id=user_id, 
        user_content_matrix=app_state["current_user_content_matrix_cf"],
        user_similarity_df=app_state["current_user_similarity_df_cf"],
        contents_df_persona=app_state["contents_df_persona"],
        all_user_personas_df=app_state["current_all_user_personas_df"],
        reactions_df=app_state["current_virtual_reactions_df"],
        reviews_df=app_state["current_virtual_reviews_df"],
        num_recommendations=10
    )

    return RecommendationResponse(
        message=f"사용자 {user_id}의 페르소나와 추천 목록이 제공되었습니다.",
        recommendations=final_recommendations_list_with_details,
        main_persona=persona_details["main_persona"],
        sub_persona=persona_details["sub_persona"],
        all_personas_scores=persona_details["all_personas_scores"]
    )


# 각 페르소나별 사용자 수 조회
@app.get("/persona-user-counts", summary="각 페르소나에 배정된 사용자 수 조회 (임시)")
async def get_persona_user_counts():
    if "current_all_user_personas_df" not in app_state or app_state["current_all_user_personas_df"].empty:
        raise HTTPException(status_code=404, detail="페르소나 데이터가 로드되지 않았거나 비어 있습니다.")

    main_personas_series = app_state["current_all_user_personas_df"].loc[
        app_state["current_all_user_personas_df"].groupby('user_id')['score'].idxmax()
    ]['persona_id']

    persona_counts = main_personas_series.value_counts().to_dict()

    return {
        "message": "각 페르소나에 배정된 사용자 수입니다.",
        "persona_user_counts": persona_counts,
        "total_users_with_persona": len(main_personas_series)
    }