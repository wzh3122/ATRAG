from typing import Optional

from sqlalchemy import select, text

from atrag.db.models import ApiKey, ApiKeyStatus
from atrag.db.repositories.base import AsyncRepositoryProtocol


class AsyncApiKeyRepositoryMixin(AsyncRepositoryProtocol):
    async def get_or_create_system_api_key(self, user: str, description: str = "atrag") -> str:
        """Return the user's active system key, creating it atomically when absent."""

        async def _operation(session):
            # PostgreSQL advisory locks also protect the empty-result case, where
            # SELECT ... FOR UPDATE has no row to lock. Other databases still get
            # transaction-level serialization where supported.
            bind = session.get_bind()
            if bind is not None and bind.dialect.name == "postgresql":
                await session.execute(
                    text("SELECT pg_advisory_xact_lock(hashtext(:lock_name))"),
                    {"lock_name": f"atrag-system-api-key:{user}"},
                )

            stmt = (
                select(ApiKey)
                .where(
                    ApiKey.user == user,
                    ApiKey.status == ApiKeyStatus.ACTIVE,
                    ApiKey.gmt_deleted.is_(None),
                    ApiKey.is_system.is_(True),
                )
                .order_by(ApiKey.gmt_created.asc())
                .limit(1)
                .with_for_update()
            )
            existing = (await session.execute(stmt)).scalars().first()
            if existing:
                return existing.key

            api_key = ApiKey(
                user=user,
                description=description,
                status=ApiKeyStatus.ACTIVE,
                is_system=True,
            )
            session.add(api_key)
            await session.flush()
            await session.refresh(api_key)
            return api_key.key

        return await self.execute_with_transaction(_operation)

    async def query_api_keys(self, user: str, is_system=False):
        """List all active API keys for a user"""

        async def _query(session):
            stmt = select(ApiKey).where(
                ApiKey.user == user,
                ApiKey.status == ApiKeyStatus.ACTIVE,
                ApiKey.gmt_deleted.is_(None),
                ApiKey.is_system == is_system,
            )
            result = await session.execute(stmt)
            return result.scalars().all()

        return await self._execute_query(_query)

    async def create_api_key(self, user: str, description: Optional[str] = None, is_system=False) -> ApiKey:
        """Create a new API key for a user"""

        async def _operation(session):
            api_key = ApiKey(user=user, description=description, status=ApiKeyStatus.ACTIVE, is_system=is_system)
            session.add(api_key)
            await session.flush()
            await session.refresh(api_key)
            return api_key

        return await self.execute_with_transaction(_operation)

    async def delete_api_key(self, user: str, key_id: str) -> bool:
        """Delete an API key (soft delete)"""

        async def _operation(session):
            stmt = select(ApiKey).where(
                ApiKey.id == key_id,
                ApiKey.user == user,
                ApiKey.status == ApiKeyStatus.ACTIVE,
                ApiKey.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            api_key = result.scalars().first()
            if not api_key:
                return None

            from datetime import datetime as dt

            api_key.status = ApiKeyStatus.DELETED
            api_key.gmt_deleted = dt.utcnow()
            session.add(api_key)
            await session.flush()
            return api_key

        return await self.execute_with_transaction(_operation)

    async def get_api_key_by_id(self, user: str, id: str) -> Optional[ApiKey]:
        """Get API key by id string"""

        async def _query(session):
            stmt = select(ApiKey).where(
                ApiKey.user == user, ApiKey.id == id, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def get_api_key_by_key(self, key: str) -> Optional[ApiKey]:
        """Get API key by key string"""

        async def _query(session):
            stmt = select(ApiKey).where(
                ApiKey.key == key, ApiKey.status == ApiKeyStatus.ACTIVE, ApiKey.gmt_deleted.is_(None)
            )
            result = await session.execute(stmt)
            return result.scalars().first()

        return await self._execute_query(_query)

    async def update_api_key_by_id(self, user: str, key_id: str, description: str) -> Optional[ApiKey]:
        """Update API key description"""

        async def _operation(session):
            stmt = select(ApiKey).where(
                ApiKey.user == user,
                ApiKey.id == key_id,
                ApiKey.status == ApiKeyStatus.ACTIVE,
                ApiKey.gmt_deleted.is_(None),
            )
            result = await session.execute(stmt)
            api_key = result.scalars().first()

            if api_key:
                api_key.description = description
                session.add(api_key)
                await session.flush()
                await session.refresh(api_key)

            return api_key

        return await self.execute_with_transaction(_operation)
