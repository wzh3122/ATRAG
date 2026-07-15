import logging
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi_users import BaseUserManager

from atrag.config import AsyncSessionDep
from atrag.db.models import Role, User
from atrag.schema import view_models
from atrag.utils.utils import utc_now

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/test/register_admin", tags=["test"])
async def test_register_admin(request: Request, data: view_models.Register, session: AsyncSessionDep):
    if os.environ.get("DEPLOYMENT_MODE") != "dev":
        raise HTTPException(status_code=403, detail="Not allowed")

    from sqlalchemy import select

    result = await session.execute(select(User).where(User.username == data.username))
    if result.scalars().first():
        raise HTTPException(status_code=400, detail="Username already exists")

    user = User(
        username=data.username,
        email=data.email,
        hashed_password=BaseUserManager(None).password_helper.hash(data.password),
        role=Role.ADMIN,
        is_active=True,
        is_verified=True,
        date_joined=utc_now(),
    )
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return view_models.User(
        id=str(user.id),
        username=user.username,
        email=user.email,
        role=user.role,
        is_active=user.is_active,
        date_joined=user.date_joined.isoformat(),
    )
