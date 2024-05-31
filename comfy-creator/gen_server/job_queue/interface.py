from abc import ABC, abstractmethod
from typing import Optional


class JobQueue(ABC):
    @abstractmethod
    def subscribe(self, topic: str, **kwargs):
        pass

    @abstractmethod
    def publish(self, topic: str, **kwargs):
        pass

    @abstractmethod
    def unsubscribe(self, topic: str, **kwargs):
        pass

    @abstractmethod
    def reader(self, topic: str, is_read_compacted: Optional[bool], **kwargs):
        pass
