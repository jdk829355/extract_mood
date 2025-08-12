# run_worker.py
import sys
import subprocess

def main():
    """
    RQ 워커를 버스트 모드로 실행하고,
    결과와 상관없이 항상 성공(exit code 0)으로 종료합니다.
    """
    print("Starting RQ worker in burst mode via wrapper...")
    # 'rq worker -b' 명령어를 실행
    subprocess.run(["rq", "worker", "-b"])
    print("Worker has finished its burst. Exiting with code 0.")
    # pm2를 위해 항상 성공 코드로 종료
    sys.exit(0)

if __name__ == '__main__':
    main()