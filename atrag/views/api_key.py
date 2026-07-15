
from fastapi import APIRouter, Depends, Request

from atrag.db.models import User
from atrag.schema.view_models import ApiKeyCreate, ApiKeyList, ApiKeyUpdate
from atrag.service.api_key_service import api_key_service
from atrag.utils.audit_decorator import audit
from atrag.views.auth import required_user

router = APIRouter()


@router.get("/apikeys", tags=["api_keys"])
async def list_api_keys_view(request: Request, user: User = Depends(required_user)) -> ApiKeyList:
    """List all API keys for the current user"""
    return await api_key_service.list_api_keys(str(user.id))


@router.post("/apikeys", tags=["api_keys"])
@audit(resource_type="api_key", api_name="CreateApiKey")
async def create_api_key_view(
    request: Request,
    api_key_create: ApiKeyCreate,
    user: User = Depends(required_user),
):
    """Create a new API key"""
    return await api_key_service.create_api_key(str(user.id), api_key_create)


@router.delete("/apikeys/{apikey_id}", tags=["api_keys"])
@audit(resource_type="api_key", api_name="DeleteApiKey")
async def delete_api_key_view(request: Request, apikey_id: str, user: User = Depends(required_user)):
    """Delete an API key"""
    return await api_key_service.delete_api_key(str(user.id), apikey_id)


@router.put("/apikeys/{apikey_id}", tags=["api_keys"])
@audit(resource_type="api_key", api_name="UpdateApiKey")
async def update_api_key_view(
    request: Request,
    apikey_id: str,
    api_key_update: ApiKeyUpdate,
    user: User = Depends(required_user),
):
    """Update an API key"""
    return await api_key_service.update_api_key(str(user.id), apikey_id, api_key_update)
