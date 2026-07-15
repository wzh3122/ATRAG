from sqlalchemy import select

from atrag.db.models import ConfigModel
from atrag.db.repositories.base import AsyncRepositoryProtocol


class AsyncSystemRepositoryMixin(AsyncRepositoryProtocol):
    async def query_config(self, key: str):
        async def _query(session):
            stmt = select(ConfigModel).where(ConfigModel.key == key)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)
