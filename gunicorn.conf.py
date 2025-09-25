import multiprocessing
import os


bind = f"0.0.0.0:{os.getenv('APP_PORT', 8000)}"

workers = multiprocessing.cpu_count() * 2 + 1

worker_class = "uvicorn.workers.UvicornWorker"

max_requests = 1000

max_requests_jitter = 50

accesslog = "-"

errorlog = "-"

loglevel = "info"

timeout=30

preload_app = False

APSC_WORKER = None

def pre_fork(server, worker):
    global APSC_WORKER
    global preload_app
    if APSC_WORKER is None and not preload_app:
        APSC_WORKER = worker
        print(f"APSC_WORKER: {APSC_WORKER.pid}")
        
def post_fork(server, worker):
    global APSC_WORKER
    os.environ["WORKER_PID"] = str(worker.pid)
    os.environ["APSC"] = "1" if worker == APSC_WORKER else "0"
    
def child_exit(server, worker):
    global APSC_WORKER
    if worker == APSC_WORKER:
        APSC_WORKER = None