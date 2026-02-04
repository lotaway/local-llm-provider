from typing import List


class ModelProvider:
    def list_models(self) -> List[str]:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError
