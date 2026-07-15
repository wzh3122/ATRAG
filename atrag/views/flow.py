from typing import Union

from fastapi import APIRouter, Depends, Request

from atrag.db.models import User
from atrag.schema.view_models import WorkflowDefinition
from atrag.service.flow_service import flow_service_global
from atrag.utils.audit_decorator import audit
from atrag.views.auth import required_user

router = APIRouter()


@router.get("/bots/{bot_id}/flow", tags=["flows"])
async def get_flow_view(
    request: Request, bot_id: str, user: User = Depends(required_user)
) -> Union[WorkflowDefinition, dict]:
    return await flow_service_global.get_flow(str(user.id), bot_id)


@router.put("/bots/{bot_id}/flow", tags=["flows"])
@audit(resource_type="flow", api_name="UpdateFlow")
async def update_flow_view(
    request: Request,
    bot_id: str,
    data: WorkflowDefinition,
    user: User = Depends(required_user),
):
    return await flow_service_global.update_flow(str(user.id), bot_id, data)
