import asyncio
import aiohttp
import os
from PIL import Image
import torch
import clip
from collections import Counter
import io
from dotenv import load_dotenv
import os
from supabase import create_client, Client

load_dotenv()

print("Loading CLIP model...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-B/32", device=DEVICE)
print(f"Model loaded on: {DEVICE}")

class AtmosphereProcessor:
    def __init__(self):
        self._load_env_vars()
        self._init_clients()
        self._load_tag_mappings()
    
    def _load_env_vars(self):
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        if not all([self.google_api_key, self.supabase_url, self.supabase_key]):
            raise ValueError("필수 환경 변수가 .env 파일에 설정되지 않았습니다.")
    def _init_clients(self):
        self.supabase: Client = create_client(self.supabase_url, self.supabase_key) # pyright: ignore[reportArgumentType]

    def _load_tag_mappings(self):
        print("Syncing mood tag info from DB...")
        try:
            response = self.supabase.table("mood_tags").select("id, mood_tag, tag_en").execute()
            self.tag_to_id = {item['mood_tag'].strip(): item['id'].strip() for item in response.data}
            self.en_to_kr = {item['tag_en'].strip(): item['mood_tag'].strip() for item in response.data}
            self.mood_labels_en = list(self.en_to_kr.keys())
            print("Tag info synced.")
        except Exception as e:
            raise RuntimeError(f"Failed to fetch tag info from DB: {e}")
    
    def _get_best_labels(self, photos, labels):
        if not photos or not labels: return []
        text_inputs=clip.tokenize(labels).to(DEVICE);image_inputs=torch.stack([PREPROCESS(p) for p in photos]).to(DEVICE)
        with torch.no_grad():
            image_features=MODEL.encode_image(image_inputs);text_features=MODEL.encode_text(text_inputs)
            image_features/=image_features.norm(dim=-1,keepdim=True);text_features/=text_features.norm(dim=-1,keepdim=True)
            similarity=(100.0*image_features@text_features.T).softmax(dim=-1);best_match_indices=similarity.argmax(dim=1).cpu().numpy()
            return[labels[i].strip() for i in best_match_indices]
        
    # 사진 한 장 다운로드 하는 함수
    # 얘를 이용해서 병렬처리 할 겨
    async def _fetch_photo(self, session: aiohttp.ClientSession, name: str) -> Image.Image | None:
        media_url = f"https://places.googleapis.com/v1/{name}/media?key={self.google_api_key}&maxHeightPx=400&skipHttpRedirect=true"
        try:
            async with session.get(media_url) as response:
                response.raise_for_status()
                data = await response.json()
                photo_uri = data.get('photoUri')
                if not photo_uri: return None
                async with session.get(photo_uri) as image_response:
                    image_response.raise_for_status()
                    image_bytes = await image_response.read()
                    return Image.open(io.BytesIO(image_bytes))
        except aiohttp.ClientError: return None
    
    async def _fetch_all_photos(self, place_id: str) -> list[Image.Image]:
        # 우선 장소 세부정보를 불러와서 사진 이름들을 받아줌
        details_url = f"https://places.googleapis.com/v1/places/{place_id}"
        headers = {'Content-Type': 'application/json', 'X-Goog-Api-Key': self.google_api_key, 'X-Goog-FieldMask': 'photos'}

        # 개별 사진 요청의 결과들이 담길 리스트
        image_results: list[Image.Image] = []

        # 장소 상세정보 불러오기
        async with aiohttp.ClientSession() as session:
            async with session.get(url=details_url, headers=headers) as response:
                if response.status != 200: return image_results
                data = await response.json()
            
            # 사진이 없는 공간이면 걍 빈 리스트 반환
            if not data.get('photos'): return image_results
            # name 필드만 추출해서 리스트로 만들기
            photo_names = [photo['name'] for photo in data['photos']]
            # 병렬처리를 위한 task 리스트
            tasks = [self._fetch_photo(session, name) for name in photo_names]
            # 한번에 실행
            downloaded_images = await asyncio.gather(*tasks)
            # 다운에 실패하면 None이기 때문에 None인 값을 걸러서 저장
            image_results = [img for img in downloaded_images if img]
        print(f"총 {len(image_results)}개의 유효한 사진을 다운로드했습니다.")
        return image_results
    
    # 레이블과 사진 리스트를 받아서 분류해줌
    # clip 모델 사용
    def _get_best_labels_for_photos(
            self,
        photos: list[Image.Image],
        labels: list[str],
    ) -> list[str]:
        """
        사진 배치와 레이블 리스트를 받아, 각 사진에 가장 적합한 레이블을 찾아 리스트로 반환합니다.
        """
        if not photos or not labels:
            return []

        # 텍스트와 이미지를 배치로 처리
        text_inputs = clip.tokenize(labels).to(DEVICE)
        image_inputs = torch.stack([PREPROCESS(p) for p in photos]).to(DEVICE) # pyright: ignore[reportArgumentType]

        with torch.no_grad():
            # 인코딩도 한 번에 수행
            image_features = MODEL.encode_image(image_inputs)
            text_features = MODEL.encode_text(text_inputs)

            # 정규화
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # 코사인 유사도 계산 및 확률 변환
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # 각 사진(row)마다 가장 높은 확률을 가진 레이블의 인덱스를 찾음
            best_match_indices = similarity.argmax(dim=1).cpu().numpy()
            
            # 인덱스를 실제 레이블 텍스트로 변환하여 반환
            return [labels[i] for i in best_match_indices]
        
    # 공간 사진만 분류해줌
    # 사진 리스트 -> 사진 리스트
    def _filter_spatial_photos(self, photos: list[Image.Image]) -> list[Image.Image]:
        print("\n[1단계] 공간 사진 필터링을 시작합니다...")
        # 사진이 이 카테고리면 괜찮은 거임
        positive_labels = ["interior", "exterior", "wide shot", "seating area"]
        # 이 카테고리면 안됨, 나쁜 사진임
        negative_labels = ["food", "drink", "person", "menu"]
        
        # 원본 사진과 해당 사진의 종류(긍정/부정)를 매칭
        photo_classifications = self._get_best_labels_for_photos(
            photos, positive_labels + negative_labels
        )
        
        # 긍정 레이블에 해당하는 사진만 필터링
        spatial_photos = [
            photo for photo, label in zip(photos, photo_classifications) if label in positive_labels
        ]
        print(f"필터링 결과: 총 {len(photos)}개 중 {len(spatial_photos)}개의 공간 사진을 선별했습니다.")
        return spatial_photos
        
    def _analyze_atmosphere(self, photos):
        # ... (이전 코드와 로직 동일)
        best_labels=self._get_best_labels_for_photos(photos,self.mood_labels_en)
        return Counter(best_labels)
    
    async def run_pipeline(self, place_id: str):
        photos = await self._fetch_all_photos(place_id)
        if not photos: return

        spatial_photos = self._filter_spatial_photos(photos)
        if not spatial_photos: return

        final_atmosphere = self._analyze_atmosphere(spatial_photos)
        top_two_moods = final_atmosphere.most_common(2)
        res = []
        for label, count in top_two_moods:
            percentage = (count / len(spatial_photos)) * 100
            res.append(self.en_to_kr[label])
            print(f"- {self.en_to_kr[label]}: {count}표 ({percentage:.1f}%)")
        
        if not res: return
        res = [x.strip() for x in res]
        print(res)
        response = (self.supabase
        .table("space_mood_tags")
        .upsert([{"mood_tag_id":self.tag_to_id[x], "space_id":place_id} for x in res])
        .execute()
        )
        return res

def extract_and_upload(place_id: str):
    asyncio.run(AtmosphereProcessor().run_pipeline(place_id))
    return