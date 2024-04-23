from abc import ABC, abstractmethod
from models.call import CallModel
from models.readiness import ReadinessStatus
from typing import Optional
from uuid import UUID


class IStore(ABC):

    @abstractmethod
    async def areadiness(self) -> ReadinessStatus:
        pass

    @abstractmethod
    async def call_aget(self, call_id: UUID) -> Optional[CallModel]:
        pass

    @abstractmethod
    async def call_aset(self, call: CallModel) -> bool:
        pass

    @abstractmethod
    async def call_asearch_one(self, phone_number: str) -> Optional[CallModel]:
        pass

    @abstractmethod
    async def call_asearch_all(self, phone_number: str) -> Optional[list[CallModel]]:
        pass
