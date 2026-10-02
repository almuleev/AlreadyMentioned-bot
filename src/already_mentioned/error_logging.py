"""Log operation context without message text, vectors, or exception details."""

import logging

logger = logging.getLogger("already_mentioned.operations")


def log_operation_error(
    operation: str, chat_id: int, message_id: int, error: Exception
) -> None:
    logger.error(
        "operation=%s chat_id=%d message_id=%d error_type=%s",
        operation,
        chat_id,
        message_id,
        type(error).__name__,
    )
