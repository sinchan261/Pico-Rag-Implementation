# import threading
# import time
# from core.queue_manager import ResultQueue

# class BackgroundWorkers:
#     def __init__(self, result_queue: ResultQueue):
#         self.result_queue = result_queue

#     def generate_image(self, prompt: str, task_id: str):
#         def task():
#             print(prompt)
#             time.sleep(3)  # Simulating image generation delay
#             result = f"Image generated for: {prompt}"
#             self.result_queue.push(task_id, result)
#         threading.Thread(target=task, daemon=True).start()


# core/workers.py
class BackgroundWorkers:
    def __init__(self, image_engine):
        self.image_engine = image_engine

    def generate_image(self, prompt: str, size: str):
        """
        This is the task function called by TaskBus.
        Returns the generated image path or failure message.
        """
        print(f"Generating image for: {prompt} with size: {size}")
        try:
            return self.image_engine.generate_image(prompt, size)
        except Exception as e:
            return f"❌ Image generation failed: {e}"
