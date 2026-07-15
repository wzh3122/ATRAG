from fastapi import APIRouter

from atrag.config import settings
from atrag.db.ops import async_db_ops
from atrag.schema.view_models import Auth, Auth0, Authing, Config, Logto
from atrag.views.utils import get_available_login_methods

router = APIRouter()


@router.get("", tags=["config"])
async def config_view() -> Config:
    auth = Auth(
        type=settings.auth_type,
    )
    match settings.auth_type:
        case "auth0":
            auth.auth0 = Auth0(
                auth_domain=settings.auth0_domain,
                auth_app_id=settings.auth0_client_id,
            )
        case "authing":
            auth.authing = Authing(
                auth_domain=settings.authing_domain,
                auth_app_id=settings.authing_app_id,
            )
        case "logto":
            auth.logto = Logto(
                auth_domain="http://" + settings.logto_domain,
                auth_app_id=settings.logto_app_id,
            )
        case "cookie":
            pass
        case _:
            raise ValueError(f"Unsupported auth type: {settings.auth_type}")

    return Config(
        auth=auth,
        admin_user_exists=await async_db_ops.query_first_user_exists(),
        login_methods=get_available_login_methods(),
    )
