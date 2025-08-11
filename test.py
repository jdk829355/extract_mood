# test.py
from redis import Redis
from rq import Queue
import tasks
import time

queue = Queue(connection=Redis())

if __name__ == "__main__":
    ids = [
        'ChIJz54Qz4JoZjURT9koiME9MyE'
    ]
    
    jobs = [queue.enqueue(tasks.extract_and_upload, place_id) for place_id in ids]
    print(f"{len(jobs)} jobs queued.")


    while not all(job.is_finished for job in jobs):
        time.sleep(1)

    for job in jobs:
        print(f"Result for {job.args[0]}: {job.result}")