
def get_model(provider: str, **kwargs):
    """Factory function to initialize and return the selected model."""
    provider = provider.lower()
    
    if provider == "videollama":
        from .video_llama import VideoLLaMAModel
        return VideoLLaMAModel(**kwargs)
    elif provider == "gemini":
        from .gemini_api import GeminiVideoModel
        return GeminiVideoModel(**kwargs)
    elif provider == "openrouter":
        from .openrouter_api import OpenRouterVideoModel
        return OpenRouterVideoModel(**kwargs)
    else:
        raise ValueError(f"Unknown model provider: {provider}")