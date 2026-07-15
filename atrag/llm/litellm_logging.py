
def setup_litellm_logging():
    import logging

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False
    logging.info("LiteLLM logging is set to WARNING level and propagation is disabled.")

    import litellm

    litellm.suppress_debug_info = True
    litellm.disable_streaming_logging = True

    # Disable callbacks to prevent LoggingWorker event loop issues in Celery tasks
    # See: https://github.com/BerriAI/litellm/issues/14521
    litellm.callbacks = []

    logging.info("LiteLLM debug info suppression is enabled and callbacks are disabled.")

    # Filter Pydantic serialization warnings globally
    import warnings

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="pydantic.*", message=".*Pydantic serializer warnings.*"
    )

    # Also filter the specific warnings we're seeing
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Expected 9 fields but got.*")

    warnings.filterwarnings("ignore", category=UserWarning, message=".*Expected.*StreamingChoices.*but got.*Choices.*")
