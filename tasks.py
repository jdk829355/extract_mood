import asyncio
import aiohttp
import os
import requests
from dotenv import load_dotenv
import os
from supabase import create_client, Client

load_dotenv()

class AtmosphereProcessor:
    def __init__(self, place_id):
        self._load_env_vars()
        self._init_clients()
        self._load_tag_mappings()
        self.supabase.table('spaces').update({'mood_tag_status': 'in-progress'}).eq('id', place_id).execute()
    
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

    
    async def _fetch_all_photos_name(self, place_id: str) -> list[str]:
        # 우선 장소 세부정보를 불러와서 사진 이름들을 받아줌
        details_url = f"https://places.googleapis.com/v1/places/{place_id}"
        headers = {'Content-Type': 'application/json', 'X-Goog-Api-Key': self.google_api_key, 'X-Goog-FieldMask': 'photos'}

        # 장소 상세정보 불러오기
        async with aiohttp.ClientSession() as session:
            async with session.get(url=details_url, headers=headers) as response:
                if response.status != 200: return []
                data = await response.json()
            
            # 사진이 없는 공간이면 걍 빈 리스트 반환
            if not data.get('photos'): return []
            # name 필드만 추출해서 리스트로 만들기
            photo_names = [photo['name'] for photo in data['photos']]
            return photo_names
    

    async def run_pipeline(self, place_id: str):
        photo_names = await self._fetch_all_photos_name(place_id)
        if not photo_names: return
        
        inference_response = requests.post("http://localhost:8000/tasks/analyze_atmosphere", json={"place_id": place_id, "photo_names": photo_names})
        if inference_response.status_code != 200: return
        inference_response = inference_response.json()
        res = inference_response.get('data')
        if not res:
            self.supabase.table('spaces').update({'mood_tag_status': 'no-tag'}).eq('id', place_id).execute()
            return  
        res = [x.strip() for x in res]
        

        response = (self.supabase
        .table("space_mood_tags")
        .upsert([{"mood_tag_id":self.tag_to_id[x], "space_id":place_id} for x in res])
        .execute()
        )
        self.supabase.table('spaces').update({'mood_tag_status': 'exist'}).eq('id', place_id).execute()
        return res

def extract_and_upload(place_id: str):
    asyncio.run(AtmosphereProcessor(place_id).run_pipeline(place_id))
    return