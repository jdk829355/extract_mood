from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from redis import Redis
from rq import Queue

# 작업 함수 import
from tasks import extract_and_upload

# FastAPI 앱 생성
app = FastAPI()

# Redis 연결 및 RQ 큐 생성
try:
    redis_conn = Redis()
    q = Queue(connection=redis_conn)
except Exception as e:
    print(f"Redis 연결 실패: {e}")

# req body 형식 만들기
class JobRequest(BaseModel):
    place_id: str

@app.post("/jobs")
def create_analysis_job(request: JobRequest):
    """
    place_id를 받아 분석 작업을 RQ 큐에 추가하는 API 엔드포인트
    """
    try:
        job = q.enqueue(extract_and_upload, request.place_id, result_ttl=3600)
        
        # 클라이언트에게 Job ID와 상태를 응답합니다.
        return {"job_id": job.id, "status": "queued"}
    except Exception as e:
        # Redis 연결 실패 등 예외 발생 시 에러 응답
        raise HTTPException(status_code=500, detail=str(e))