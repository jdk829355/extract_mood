import os
import clip
from dotenv import load_dotenv
from supabase import create_client
import torch
from PIL import Image


print("CLIP 모델을 로딩합니다...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(f"CLIP 모델 로딩 완료. Device: {device}")

load_dotenv()
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase = create_client(supabase_url, supabase_key) # type: ignore

path = "./wallpapers/categorized"
file_list = os.listdir(path)

category_mapper = {
    "arch": "architecture",
    "art": "art",
    "business":"business",
    'coding':'coding',
    'hum':'humanity',
    'nature':'nature',
    'sports': "sports",
    'tech':'technology',
    'writing':'writing',
    'space':'space'
}

text_inputs = clip.tokenize([category_mapper[file.split('_')[1]] for file in file_list]).to(device)

with torch.no_grad():
    image_features = model.encode_text(text_inputs)
    image_features /= image_features.norm(dim=-1, keepdim=True)

image_features = image_features.tolist()

data = [{'path': "categorized/"+file_list[i], 'feature': image_features[i]} for i in range(len(file_list))]

supabase.table("study_wallpaper").upsert(data).execute()