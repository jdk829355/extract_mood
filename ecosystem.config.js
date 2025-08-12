module.exports = {
  apps : [{
    name   : "uvicorn-server",
    script : "uvicorn", // 실행할 명령어
    args   : "server:app --host 0.0.0.0 --port 8000", // 명령어에 전달할 인자
    watch  : ["server.py", "tasks.py", ".env"], // 이 파일들이 변경되면 자동 재시작
    interpreter: "./venv/bin/python", // 가상환경의 파이썬 사용
    env: {
      "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES"
    }
  },
  {
    name: "rq",
    script:"rq",
    args: "worker",
    interpreter: "./venv/bin/python",
    watch  : false,
    autorestart: true,
    env: {
      "OBJC_DISABLE_INITIALIZE_FORK_SAFETY": "YES", // macOS용 환경 변수
    }
  }
]
};
