"""
장소 분위기 분석 관련 코드는 출시 전까지 쓸 일 없으므로 주석처리
배경 추천만 활성화
"""


# import asyncio
# from collections import Counter
# import aiohttp
import clip
import deepl
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from rq import Queue
from supabase import Client, create_client
import torch
# import io
# from PIL import Image
import os
from contextlib import asynccontextmanager
import random
from transformers import pipeline

# 작업 함수 import
# from tasks import extract_and_upload

server_state= {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("--- 서버 시작: 리소스를 로딩합니다. ---")
    load_dotenv()
    # 1. RQ 큐 연결
    redis_conn = Redis()
    server_state["rq_queue"] = Queue(connection=redis_conn)
    print("RQ 큐 연결 완료.")
    
    # # 2. CLIP 모델 로딩 (가장 오래 걸리는 작업)
    # print("CLIP 모델을 로딩합니다...")
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, preprocess = clip.load("RN101", device=device)
    # server_state["clip_model"] = model
    # server_state["clip_preprocess"] = preprocess
    # server_state["device"] = device
    # print(f"CLIP 모델 로딩 완료. Device: {device}")

    # 제로샷 분류 파이프라인 로드
    print("제로샷 분류 파이프라인을 로딩합니다...")
    server_state['classifier'] = pipeline("zero-shot-classification", 
                        model="facebook/bart-large-mnli")
    print("제로샷 분류 파이프라인 로딩 완료.")

    yield
    print("--- 서버 종료: 리소스를 해제합니다. ---")
    server_state.clear()
# class AtmosphereProcessor:
#     def __init__(self):
#         self._load_env_vars()
#         self._init_clients()
#         self._load_tag_mappings()

    

#     def _load_env_vars(self):
#         self.google_api_key = os.getenv("GOOGLE_API_KEY")
#         self.supabase_url = os.getenv("SUPABASE_URL")
#         self.supabase_key = os.getenv("SUPABASE_KEY")
#         if not all([self.google_api_key, self.supabase_url, self.supabase_key]):
#             raise ValueError("필수 환경 변수가 .env 파일에 설정되지 않았습니다.")
    
#     def _init_clients(self):
#         self.supabase: Client = create_client(self.supabase_url, self.supabase_key) # pyright: ignore[reportArgumentType]
    
#     def _load_tag_mappings(self):
#         print("Syncing mood tag info from DB...")
#         try:
#             response = self.supabase.table("mood_tags").select("id, mood_tag, tag_en").execute()
#             self.tag_to_id = {item['mood_tag'].strip(): item['id'].strip() for item in response.data}
#             self.en_to_kr = {item['tag_en'].strip(): item['mood_tag'].strip() for item in response.data}
#             self.mood_labels_en = list(self.en_to_kr.keys())
#             print("Tag info synced.")
#         except Exception as e:
#             raise RuntimeError(f"Failed to fetch tag info from DB: {e}")
    
#     async def _fetch_photo(self, session: aiohttp.ClientSession, name: str) -> Image.Image | None:
#         media_url = f"https://places.googleapis.com/v1/{name}/media?key={self.google_api_key}&maxHeightPx=400&skipHttpRedirect=true"
#         try:
#             async with session.get(media_url) as response:
#                 response.raise_for_status()
#                 data = await response.json()
#                 photo_uri = data.get('photoUri')
#                 if not photo_uri: return None
#                 async with session.get(photo_uri) as image_response:
#                     image_response.raise_for_status()
#                     image_bytes = await image_response.read()
#                     return Image.open(io.BytesIO(image_bytes))
#         except aiohttp.ClientError: return None
    
#     async def _fetch_all_photos(self, photo_names: list[str]) -> list[Image.Image]:
#         # 개별 사진 요청의 결과들이 담길 리스트
#         image_results: list[Image.Image] = []

#         async with aiohttp.ClientSession() as session:
#             # 병렬처리를 위한 task 리스트
#             tasks = [self._fetch_photo(session, name) for name in photo_names]
#             # 한번에 실행
#             downloaded_images = await asyncio.gather(*tasks)
#             # 다운에 실패하면 None이기 때문에 None인 값을 걸러서 저장
#             image_results = [img for img in downloaded_images if img]
#         print(f"총 {len(image_results)}개의 유효한 사진을 다운로드했습니다.")
#         return image_results
    
#     # 레이블과 사진 리스트를 받아서 분류해줌
#     # clip 모델 사용
#     def _get_best_labels_for_photos(
#             self,
#         photos: list[Image.Image],
#         labels: list[str],
#     ) -> list[str]:
#         """
#         사진 배치와 레이블 리스트를 받아, 각 사진에 가장 적합한 레이블을 찾아 리스트로 반환합니다.
#         """
#         if not photos or not labels:
#             return []

#         # 텍스트와 이미지를 배치로 처리
#         text_inputs = clip.tokenize(labels).to(server_state["device"])
#         image_inputs = torch.stack([server_state["clip_preprocess"](p) for p in photos]).to(server_state["device"]) # pyright: ignore[reportArgumentType]

#         with torch.no_grad():
#             # 인코딩도 한 번에 수행
#             image_features = server_state["clip_model"].encode_image(image_inputs)
#             text_features = server_state["clip_model"].encode_text(text_inputs)

#             # 정규화
#             image_features /= image_features.norm(dim=-1, keepdim=True)
#             text_features /= text_features.norm(dim=-1, keepdim=True)

#             # 코사인 유사도 계산 및 확률 변환
#             similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
#             # 각 사진(row)마다 가장 높은 확률을 가진 레이블의 인덱스를 찾음
#             best_match_indices = similarity.argmax(dim=1).cpu().numpy()
            
#             # 인덱스를 실제 레이블 텍스트로 변환하여 반환
#             return [labels[i] for i in best_match_indices]
    
#     # 공간 사진만 분류해줌
#     # 사진 리스트 -> 사진 리스트
#     def _filter_spatial_photos(self, photos: list[Image.Image]) -> list[Image.Image]:
#         print("\n[1단계] 공간 사진 필터링을 시작합니다...")
#         # 사진이 이 카테고리면 괜찮은 거임
#         positive_labels = ["interior", "exterior", "wide shot", "seating area"]
#         # 이 카테고리면 안됨, 나쁜 사진임
#         negative_labels = ["food", "drink", "person", "menu"]
        
#         # 원본 사진과 해당 사진의 종류(긍정/부정)를 매칭
#         photo_classifications = self._get_best_labels_for_photos(
#             photos, positive_labels + negative_labels
#         )
        
#         # 긍정 레이블에 해당하는 사진만 필터링
#         spatial_photos = [
#             photo for photo, label in zip(photos, photo_classifications) if label in positive_labels
#         ]
#         print(f"필터링 결과: 총 {len(photos)}개 중 {len(spatial_photos)}개의 공간 사진을 선별했습니다.")
#         return spatial_photos
    
#     def _analyze_atmosphere(self, photos):
#         best_labels=self._get_best_labels_for_photos(photos,self.mood_labels_en)
#         return Counter(best_labels)
    
# 공부 세션 중 배경화면 선정 로직 (제목에 따라 image retrieval)
class BackgroundPicker:
    def __init__(self):
        self._load_env_vars()
        self._init_clients()

    def _load_env_vars(self):
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        self.deepl_key = os.getenv("DEEPL_KEY")
        if not all([self.supabase_url, self.supabase_key]):
            raise ValueError
        
    def _init_clients(self):
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key) # pyright: ignore[reportArgumentType]
    
    def _get_best_photo_for_title(self, title:str):
        deepl_client = deepl.DeepLClient(self.deepl_key) # type: ignore
        title = deepl_client.translate_text(title, target_lang="EN-US").text # type: ignore

        category_res = self.supabase.table('wallpaper_category').select('*').execute()
        category = [x['category'] for x in category_res.data]

        # 분류 실행
        result = server_state['classifier'](title, category)
        print(f"분류 결과: {result['labels'][0]}, {result['scores'][0]}")
        

        path = ""
        if(result['scores'][0] < 0.3):
            general_images = self.supabase.storage.from_('wallpaper').list(
                "general",{
                    "limit": 100,
                    "offset": 0,
                    "sortBy": {"column": "name", "order": "desc"},
            })
            # 랜덤으로 하나 뽑음
            path =  random.sample(['general/'+x['name'] for x in general_images], 1)[0]
        else:
            category_images = self.supabase.table('study_wallpaper').select('*').eq('category', result['labels'][0]).execute()
            path = random.sample(category_images.data, 1)[0]['path']

        response = (self.supabase.storage
        .from_("wallpaper")
        .create_signed_url(
            path,
            60*60*24*7, # 1주일
        ))
        return response['signedUrl']


# FastAPI 앱 생성
app = FastAPI(lifespan=lifespan)

# Redis 연결 및 RQ 큐 생성
try:
    redis_conn = Redis()
    q = Queue(connection=redis_conn)
except Exception as e:
    print(f"Redis 연결 실패: {e}")

# req body 형식 만들기
class JobRequest(BaseModel):
    place_id: str

class WallpaperRequest(BaseModel):
    title: str

class AtmosphereRequest(BaseModel):
    place_id: str
    photo_names: list[str]

# atmosphere_processor = AtmosphereProcessor()



# @app.post("/jobs")
# def create_analysis_job(request: JobRequest):
#     """
#     place_id를 받아 분석 작업을 RQ 큐에 추가하는 API 엔드포인트
#     """
#     try:
#         job = q.enqueue(extract_and_upload, request.place_id, result_ttl=3600)
#         supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY")) # pyright: ignore[reportArgumentType]
#         supabase.table('spaces').update({'mood_tag_status': 'queued'}).eq('id', request.place_id).execute()
        
#         # 클라이언트에게 Job ID와 상태를 응답합니다.
#         return {"job_id": job.id, "status": "queued"}
#     except Exception as e:
#         # Redis 연결 실패 등 예외 발생 시 에러 응답
#         raise HTTPException(status_code=500, detail=str(e))
    

    
# @app.post("/tasks/analyze_atmosphere")
# async def analyze_atmosphere(request: AtmosphereRequest):
#     photos = await atmosphere_processor._fetch_all_photos(request.photo_names)
#     if not photos: return
    
#     spatial_photos = atmosphere_processor._filter_spatial_photos(photos)
#     if not spatial_photos: return
    
#     final_atmosphere = atmosphere_processor._analyze_atmosphere(spatial_photos)
#     top_two_moods = final_atmosphere.most_common(2)
#     res = []
#     for label, count in top_two_moods:
#         percentage = (count / len(spatial_photos)) * 100
#         res.append(atmosphere_processor.en_to_kr[label])
#         print(f"- {atmosphere_processor.en_to_kr[label]}: {count}표 ({percentage:.1f}%)")
    
#     if not res: return
#     res = [x.strip() for x in res]
#     data = {
#         'data':res
#     }
#     return data

@app.post("/tasks/wallpaper")
def get_wallpaper(request: WallpaperRequest):
    wallpaper_picker = BackgroundPicker()
    response = wallpaper_picker._get_best_photo_for_title(request.title)
    return {"url":response}