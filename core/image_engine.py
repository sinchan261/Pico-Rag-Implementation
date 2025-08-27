import requests
import os

class ImageEngine:
    def __init__(self, save_path="generated_images"):
        self.save_path = save_path
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def generate_image(self, prompt: str, size: str = "medium") -> str:
        """
        Generate an image using Pollinations API and save it locally.
        Returns the file path.
        """
        url = f"https://image.pollinations.ai/prompt/{prompt.replace(' ', '%20')}"
        response = requests.get(url)

        if response.status_code == 200:
            file_name = f"{prompt[:20].replace(' ', '_')}_{size}.png"
            file_path = os.path.join(self.save_path, file_name)
            with open(file_path, "wb") as f:
                f.write(response.content)
            return file_path
        else:
            return "⚠️ Failed to generate image."
