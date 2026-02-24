from abc import ABC, abstractmethod

class BaseVideoModel(ABC):
    @abstractmethod
    def analyze_video(self, video_path: str, prompt: str) -> str:
        """
        Takes a video path and a text prompt, and returns the model's text response.
        """
        pass