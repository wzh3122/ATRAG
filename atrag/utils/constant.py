KEY_USER_ID = "X-USER-ID"
KEY_BOT_ID = "X-BOT-ID"
KEY_CHAT_ID = "X-CHAT-ID"
KEY_WEBSOCKET_PROTOCOL = "Sec-Websocket-Protocol"
DOC_QA_REFERENCES = "|DOC_QA_REFERENCES|"
DOCUMENT_URLS = "|DOCUMENT_URLS|"
class QuotaType:
    MAX_BOT_COUNT = "max_bot_count"
    MAX_COLLECTION_COUNT = "max_collection_count"
    MAX_DOCUMENT_COUNT = "max_document_count"
    MAX_DOCUMENT_COUNT_PER_COLLECTION = "max_document_count_per_collection"
    MAX_CONVERSATION_COUNT = "max_conversation_count"


class IndexAction:
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
