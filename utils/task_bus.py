# utils/task_bus.py
import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class TaskResult:
    task_id: str
    kind: str            # e.g. "image_done"
    message: str         # human-friendly one-liner
    payload: Optional[Any] = None  # extra data (e.g., file path)


class TaskBus:
    """
    A tiny dispatcher for background jobs.
    - submit(func, *args, kind="...", done_msg="..."): returns task_id
    - results(): non-blocking read of finished results (Queue)
    """
    def __init__(self):
        self._results = queue.Queue()
        self._tasks = {}  # task_id -> {"cancel": threading.Event}
    
    def submit(self, func: Callable, *args, kind: str, done_msg: str) -> str:
        task_id = str(uuid.uuid4())
        cancel_event = threading.Event()
        self._tasks[task_id] = {"cancel": cancel_event}

        def runner():
            payload = None
            try:
                payload = func(*args)  # your function returns something (e.g., file path)
                # only push result if not cancelled
                if not cancel_event.is_set():
                    self._results.put(TaskResult(task_id, kind, done_msg, payload))
            except Exception as e:
                if not cancel_event.is_set():
                    self._results.put(TaskResult(task_id, kind, f"❌ Task failed: {e}", None))
            finally:
                # best-effort cleanup
                self._tasks.pop(task_id, None)

        threading.Thread(target=runner, daemon=True).start()
        return task_id

    def cancel(self, task_id: str) -> bool:
        meta = self._tasks.get(task_id)
        if not meta:
            return False
        meta["cancel"].set()
        return True

    def has_result(self) -> bool:
        return not self._results.empty()

    def get_result(self) -> Optional[TaskResult]:
        try:
            return self._results.get_nowait()
        except queue.Empty:
            return None
