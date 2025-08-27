import queue

class ResultQueue:
    def __init__(self):
        self.queue = queue.Queue()

    def push(self, task_id, result):
        self.queue.put((task_id, result))

    def pop(self):
        if not self.queue.empty():
            return self.queue.get()
        return None
