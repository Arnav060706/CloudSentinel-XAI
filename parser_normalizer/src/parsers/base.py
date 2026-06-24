from abc import ABC, abstractmethod
from src.schema import UnifiedLogModel

class BaseParser(ABC):
    @abstractmethod
    def parse(self, raw_log: dict) -> dict:
        """Parses a raw log dict and returns a dict matching the Unified Schema."""
        pass
