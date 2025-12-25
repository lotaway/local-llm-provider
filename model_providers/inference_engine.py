from abc import ABC, abstractmethod
from typing import AsyncGenerator, Any, cast


class InferenceEngine(ABC):
    @abstractmethod
    async def generate_stream(self, prompt: str, **kwargs) -> AsyncGenerator[str, None]:
        pass

    @abstractmethod
    def unload(self):
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass
