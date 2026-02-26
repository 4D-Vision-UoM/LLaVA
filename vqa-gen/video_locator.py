from pathlib import Path

class VideoLocator:
    """Retrieves videos from an external directory structured by splits."""
    def __init__(self, external_videos_base_dir: str):
        self.base_dir = Path(external_videos_base_dir)

    def find_video(self, sequence_name: str, split: str = "train") -> Path:
        """
        Looks for the video matching the sequence name in the designated split folder.
        Checks common video extensions.
        """
        split_dir = self.base_dir / split
        if split == "train":
            split_dir = split_dir / "Env1"  # Ensure we look in the correct split folder
        if not split_dir.exists():
            return None
            
        # Search for common extensions for the given sequence
        for ext in ['.mp4', '.webm', '.avi']:
            target_path = split_dir / f"{sequence_name}{ext}"
            if target_path.exists():
                return target_path
        return None