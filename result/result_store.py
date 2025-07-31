from typing import Dict
from threading import Lock

result_data: Dict[str, dict] = {}
result_lock = Lock()

def save_result(uuid: str, data: dict):
    with result_lock:
        result_data[uuid] = {"status": "done", "data": data}

def get_result(uuid: str):
    with result_lock:
        return result_data.get(uuid, {"status": "processing"})

def init_result(uuid: str):
    with result_lock:
        result_data[uuid] = {"status": "processing"}