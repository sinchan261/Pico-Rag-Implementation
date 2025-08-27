# from core.workers import BackgroundWorkers

# class CommandDispatcher:
#     def __init__(self, workers: BackgroundWorkers):
#         self.workers = workers

#     def handle(self, user_input: str, task_id: str):
#         if user_input.lower().startswith("generate image"):
#             prompt = user_input.replace("generate image", "").strip()
#             self.workers.generate_image(prompt, task_id)
#             return "Okay, I will generate your image in the background."
#         return None


# core/dispatcher.py
from core.workers import BackgroundWorkers

class CommandDispatcher:
    def __init__(self, workers: BackgroundWorkers, task_bus, default_size: str = "1024x1024"):
        self.workers = workers
        self.task_bus = task_bus
        self.default_size = default_size

    def handle(self, description: str) -> str:
        """
        Submits the background image generation task to TaskBus.
        Returns task_id.
        """
        task_id = self.task_bus.submit(
            self.workers.generate_image,
            description,
            self.default_size,
            kind="image_done",
            done_msg="Your image is ready."
        )
        return task_id
