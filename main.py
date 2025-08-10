import asyncio
import aiohttp
import os
from PIL import Image
import torch
import clip
from collections import Counter
import io
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- 사진 다운로드 함수 (이전과 동일) ---
async def fetch_photo(session: aiohttp.ClientSession, name: str) -> Image.Image | None:
    media_url = f"https://places.googleapis.com/v1/{name}/media?key={API_KEY}&maxHeightPx=400&skipHttpRedirect=true"
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

async def fetch_all_photos(place_id: str) -> list[Image.Image]:
    details_url = f"https://places.googleapis.com/v1/places/{place_id}"
    headers = {'Content-Type': 'application/json', 'X-Goog-Api-Key': API_KEY, 'X-Goog-FieldMask': 'photos'}
    image_results: list[Image.Image] = []
    async with aiohttp.ClientSession() as session:
        async with session.get(url=details_url, headers=headers) as response:
            if response.status != 200: return image_results
            data = await response.json()

        if not data.get('photos'): return image_results
        photo_names = [photo['name'] for photo in data['photos']]
        tasks = [fetch_photo(session, name) for name in photo_names]
        downloaded_images = await asyncio.gather(*tasks)
        image_results = [img for img in downloaded_images if img]
    print(f"총 {len(image_results)}개의 유효한 사진을 다운로드했습니다.")
    return image_results

# --- 리팩토링된 핵심 분석 유틸리티 함수 ---
def get_best_labels_for_photos(
    photos: list[Image.Image],
    labels: list[str],
    model,
    preprocess,
    device: str
) -> list[str]:
    """
    사진 배치와 레이블 리스트를 받아, 각 사진에 가장 적합한 레이블을 찾아 리스트로 반환합니다.
    """
    if not photos or not labels:
        return []

    # 텍스트와 이미지를 배치로 처리
    text_inputs = clip.tokenize(labels).to(device)
    image_inputs = torch.stack([preprocess(p) for p in photos]).to(device)

    with torch.no_grad():
        # 인코딩도 한 번에 수행
        image_features = model.encode_image(image_inputs)
        text_features = model.encode_text(text_inputs)

        # 정규화
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # 코사인 유사도 계산 및 확률 변환
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # 각 사진(row)마다 가장 높은 확률을 가진 레이블의 인덱스를 찾음
        best_match_indices = similarity.argmax(dim=1).cpu().numpy()
        
        # 인덱스를 실제 레이블 텍스트로 변환하여 반환
        return [labels[i] for i in best_match_indices]

# --- 새로운 유틸리티 함수를 사용하는 간결해진 함수들 ---
def filter_spatial_photos(photos: list[Image.Image], model, preprocess, device: str) -> list[Image.Image]:
    print("\n[1단계] 공간 사진 필터링을 시작합니다...")
    positive_labels = ["interior", "exterior", "wide shot", "seating area"]
    negative_labels = ["food", "drink", "person", "menu"]
    
    # 원본 사진과 해당 사진의 종류(긍정/부정)를 매칭
    photo_classifications = get_best_labels_for_photos(
        photos, positive_labels + negative_labels, model, preprocess, device
    )
    
    # 긍정 레이블에 해당하는 사진만 필터링
    spatial_photos = [
        photo for photo, label in zip(photos, photo_classifications) if label in positive_labels
    ]
    print(f"필터링 결과: 총 {len(photos)}개 중 {len(spatial_photos)}개의 공간 사진을 선별했습니다.")
    return spatial_photos

def analyze_place_atmosphere(photos: list[Image.Image], labels: list[str], model, preprocess, device: str) -> Counter:
    print("\n[2단계] 선별된 사진으로 분위기 분석을 시작합니다...")
    best_labels = get_best_labels_for_photos(photos, labels, model, preprocess, device)
    # 각 사진의 최고 분위기 레이블을 카운트하여 반환
    return Counter(best_labels)

async def main():
    # --- 설정 부분 (이전과 동일) ---
    place_id = "ChIJPyUeoeiifDURWsRVI6DoUow"
    mood_labels_en = [
        "Quiet", "Cozy", "Aesthetic", "Lively", "Spacious",
        "Nature-Inspired", "Conceptual", "Trendy", "Modern"
    ]
    mood_tags_kr = {
        "Quiet": "조용한", "Cozy": "아늑한", "Aesthetic": "감성적인", "Lively": "활기찬",
        "Spacious": "넓고 개방적인", "Nature-Inspired": "자연 친화적인", "Conceptual": "컨셉 있는",
        "Trendy": "트렌디한", "Modern": "현대적인"
    }

    # --- 실행 부분 ---
    # 1. 사진 다운로드
    all_photos = await fetch_all_photos(place_id)
    if not all_photos: return

    # 2. 모델 로딩
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    # 3. 공간 사진 필터링
    spatial_photos = filter_spatial_photos(all_photos, model, preprocess, device)
    if not spatial_photos: return

    # 4. 분위기 분석
    final_atmosphere = analyze_place_atmosphere(spatial_photos, mood_labels_en, model, preprocess, device)

    # 5. 최종 결과 출력 (상위 2개 태그 선택)
    print("\n--- 최종 분석 결과 ---")
    if not final_atmosphere:
        print("분석 결과가 없습니다.")
        return
    
    top_two_moods = final_atmosphere.most_common(2)
    res = []
    for label, count in top_two_moods:
        percentage = (count / len(spatial_photos)) * 100
        res.append(mood_tags_kr[label])
        print(f"- {mood_tags_kr[label]}: {count}표 ({percentage:.1f}%)")
    return res


if __name__ == "__main__":
    # .env 파일이 없거나 API_KEY가 설정되지 않은 경우를 위한 확인
    if not API_KEY:
        print("'.env' 파일에 GOOGLE_API_KEY를 설정해주세요.")
    else:
        asyncio.run(main())