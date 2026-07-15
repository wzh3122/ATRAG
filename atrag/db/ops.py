import logging

from atrag.db.repositories.api_key import AsyncApiKeyRepositoryMixin
from atrag.db.repositories.base import AsyncBaseRepository, SyncBaseRepository
from atrag.db.repositories.bot import AsyncBotRepositoryMixin
from atrag.db.repositories.chat import AsyncChatRepositoryMixin
from atrag.db.repositories.collection import (
    AsyncCollectionRepositoryMixin,
    CollectionRepositoryMixin,
)
from atrag.db.repositories.document import (
    AsyncDocumentRepositoryMixin,
    DocumentRepositoryMixin,
)
from atrag.db.repositories.document_index import AsyncDocumentIndexRepositoryMixin
from atrag.db.repositories.evaluation import AsyncEvaluationRepositoryMixin
from atrag.db.repositories.graph import GraphRepositoryMixin
from atrag.db.repositories.lightrag import LightragRepositoryMixin
from atrag.db.repositories.llm_provider import (
    AsyncLlmProviderRepositoryMixin,
    LlmProviderRepositoryMixin,
)
from atrag.db.repositories.marketplace import AsyncMarketplaceRepositoryMixin
from atrag.db.repositories.marketplace_collection import AsyncMarketplaceCollectionRepositoryMixin
from atrag.db.repositories.merge_suggestion import MergeSuggestionRepository
from atrag.db.repositories.prompt_template import AsyncPromptTemplateRepositoryMixin
from atrag.db.repositories.question_set import AsyncQuestionSetRepositoryMixin
from atrag.db.repositories.search import AsyncSearchRepositoryMixin
from atrag.db.repositories.setting import (
    AsyncSettingRepositoryMixin,
    SettingRepositoryMixin,
)
from atrag.db.repositories.system import AsyncSystemRepositoryMixin
from atrag.db.repositories.user import AsyncUserRepositoryMixin

logger = logging.getLogger(__name__)


class DatabaseOps(
    SyncBaseRepository,
    CollectionRepositoryMixin,
    DocumentRepositoryMixin,
    LlmProviderRepositoryMixin,
    LightragRepositoryMixin,
    GraphRepositoryMixin,
    SettingRepositoryMixin,
):
    pass


class AsyncDatabaseOps(
    AsyncBaseRepository,
    AsyncApiKeyRepositoryMixin,
    AsyncCollectionRepositoryMixin,
    AsyncDocumentRepositoryMixin,
    AsyncBotRepositoryMixin,
    AsyncChatRepositoryMixin,
    AsyncUserRepositoryMixin,
    AsyncLlmProviderRepositoryMixin,
    AsyncMarketplaceRepositoryMixin,
    AsyncMarketplaceCollectionRepositoryMixin,
    AsyncSystemRepositoryMixin,
    AsyncSearchRepositoryMixin,
    MergeSuggestionRepository,
    AsyncDocumentIndexRepositoryMixin,
    AsyncSettingRepositoryMixin,
    AsyncPromptTemplateRepositoryMixin,
    AsyncEvaluationRepositoryMixin,
    AsyncQuestionSetRepositoryMixin,
):
    pass


async_db_ops = AsyncDatabaseOps()
db_ops = DatabaseOps()
