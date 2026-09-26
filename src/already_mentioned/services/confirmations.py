"""Short-lived administrator confirmations for /forget."""

import secrets
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True, slots=True)
class ForgetRequest:
    chat_id: int
    user_id: int
    expires_at: float


class ForgetConfirmations:
    def __init__(
        self, ttl_seconds: int = 5 * 60, clock: Callable[[], float] = monotonic
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.clock = clock
        self._requests: dict[str, ForgetRequest] = {}

    def create(self, chat_id: int, user_id: int) -> str:
        self._remove_expired()
        token = secrets.token_urlsafe(8)
        self._requests[token] = ForgetRequest(
            chat_id=chat_id,
            user_id=user_id,
            expires_at=self.clock() + self.ttl_seconds,
        )
        return token

    def consume(self, token: str, chat_id: int, user_id: int) -> bool:
        self._remove_expired()
        request = self._requests.get(token)
        if request is None or (request.chat_id, request.user_id) != (chat_id, user_id):
            return False
        del self._requests[token]
        return True

    def _remove_expired(self) -> None:
        now = self.clock()
        for token, request in list(self._requests.items()):
            if request.expires_at <= now:
                del self._requests[token]
