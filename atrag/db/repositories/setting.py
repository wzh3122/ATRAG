from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from atrag.db import models as db_models
from atrag.db.repositories.base import AsyncRepositoryProtocol, SyncRepositoryProtocol


class SettingRepositoryMixin(SyncRepositoryProtocol):
    def query_all_settings(self) -> list[db_models.Setting]:
        def _query(session):
            stmt = select(db_models.Setting)
            result = session.execute(stmt)
            return result.scalars().all()

        return self._execute_query(_query)


class AsyncSettingRepositoryMixin(AsyncRepositoryProtocol):
    async def query_setting(self, key: str) -> db_models.Setting | None:
        async def _query(session):
            stmt = select(db_models.Setting).where(db_models.Setting.key == key)
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def query_all_settings(self) -> list[db_models.Setting]:
        async def _query(session):
            stmt = select(db_models.Setting)
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def update_setting(self, key: str, value: str):
        async def _operation(session):
            stmt = (
                insert(db_models.Setting)
                .values(key=key, value=value)
                .on_conflict_do_update(
                    index_elements=["key"],
                    set_=dict(value=value),
                )
            )
            await session.execute(stmt)

        await self.execute_with_transaction(_operation)
