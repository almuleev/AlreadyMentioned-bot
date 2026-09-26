"""Short-lived record of admin replies, without storing chat history in SQLite."""

from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True, slots=True)
class PendingAnswer:
    chat_id: int
    answer_message_id: int
    answer_author_id: int
    question_message_id: int
    question_text: str
    question_link: str | None
    recorded_at: float


class ReplyCache:
    """Keep only reply chains that could later be confirmed with /solve."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 6 * 60 * 60,
        max_entries: int = 5000,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if ttl_seconds <= 0 or max_entries <= 0:
            raise ValueError("Cache limits must be positive")
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.clock = clock
        self._items: OrderedDict[tuple[int, int], PendingAnswer] = OrderedDict()

    def remember(self, answer: PendingAnswer) -> None:
        self._discard_expired()
        key = (answer.chat_id, answer.answer_message_id)
        self._items[key] = answer
        self._items.move_to_end(key)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)

    def get(self, chat_id: int, answer_message_id: int) -> PendingAnswer | None:
        self._discard_expired()
        return self._items.get((chat_id, answer_message_id))

    def clear_chat(self, chat_id: int) -> None:
        for key in list(self._items):
            if key[0] == chat_id:
                del self._items[key]

    def _discard_expired(self) -> None:
        cutoff = self.clock() - self.ttl_seconds
        while self._items:
            first_key = next(iter(self._items))
            if self._items[first_key].recorded_at > cutoff:
                break
            self._items.popitem(last=False)
