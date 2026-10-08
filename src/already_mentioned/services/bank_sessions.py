"""Bounded, expiring bank search, edit prompts and deletion previews."""

import secrets
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic

from already_mentioned.models.entities import SolutionRevision


@dataclass(frozen=True, slots=True)
class BankState:
    chat_id: int
    user_id: int
    menu_id: int
    expires_at: float
    kind: str
    view: str = "-"
    page: int = 1
    query: str = ""
    revision: SolutionRevision | None = None


class BankSessions:
    def __init__(self, *, clock=monotonic, ttl=300, capacity=500):
        if ttl <= 0 or capacity <= 0:
            raise ValueError("Cache limits must be positive")
        self.clock, self.ttl, self.capacity = clock, ttl, capacity
        self.inputs, self.views, self.deletions = (
            OrderedDict(),
            OrderedDict(),
            OrderedDict(),
        )
        self.processing = set()

    def _clean(self, entries):
        for key, state in list(entries.items()):
            if state.expires_at <= self.clock():
                del entries[key]
        while len(entries) > self.capacity:
            entries.popitem(last=False)

    def _put(self, entries, key, chat_id, user_id, menu_id, kind, **kwargs):
        entries[key] = BankState(
            chat_id, user_id, menu_id, self.clock() + self.ttl, kind, **kwargs
        )
        self._clean(entries)
        return key

    def _get(self, entries, key, chat_id, user_id, menu_id, *, consume=False):
        self._clean(entries)
        state = entries.get(key)
        if state and (state.chat_id, state.user_id, state.menu_id) == (
            chat_id,
            user_id,
            menu_id,
        ):
            if consume:
                del entries[key]
            return state
        return None

    def query(self, chat_id, user_id, menu_id, text):
        return self._put(
            self.views,
            secrets.token_urlsafe(8),
            chat_id,
            user_id,
            menu_id,
            "search",
            query=text,
        )

    def view(self, token, chat_id, user_id, menu_id):
        return self._get(self.views, token, chat_id, user_id, menu_id)

    def add_input(self, prompt_id, chat_id, user_id, menu_id, kind, **kwargs):
        self._put(
            self.inputs, (chat_id, prompt_id), chat_id, user_id, menu_id, kind, **kwargs
        )

    def input(self, chat_id, prompt_id):
        self._clean(self.inputs)
        return self.inputs.get((chat_id, prompt_id))

    def claim_input(self, prompt_id, chat_id, user_id, menu_id):
        state = self._get(self.inputs, (chat_id, prompt_id), chat_id, user_id, menu_id)
        self.processing.intersection_update(self.inputs)
        key = (chat_id, prompt_id)
        if state is None or key in self.processing:
            return None
        self.processing.add(key)
        return state

    def input_is_active(self, prompt_id, state):
        return self.input(state.chat_id, prompt_id) == state

    def finish_input(self, chat_id, prompt_id):
        self.inputs.pop((chat_id, prompt_id), None)
        self.processing.discard((chat_id, prompt_id))

    def pending(self, chat_id, user_id, menu_id, kind, solution_id=None):
        self._clean(self.inputs)
        return any(
            (s.chat_id, s.user_id, s.menu_id, s.kind)
            == (chat_id, user_id, menu_id, kind)
            and (s.revision.id if s.revision else None) == solution_id
            for s in self.inputs.values()
        )

    def deletion(self, chat_id, user_id, menu_id, revision, view, page):
        return self._put(
            self.deletions,
            secrets.token_urlsafe(8),
            chat_id,
            user_id,
            menu_id,
            "delete",
            revision=revision,
            view=view,
            page=page,
        )

    def take_deletion(self, token, chat_id, user_id, menu_id):
        return self._get(self.deletions, token, chat_id, user_id, menu_id, consume=True)

    def discard_menu(self, chat_id, menu_id):
        # Search views survive card navigation; pending writes do not.
        for entries in (self.inputs, self.deletions):
            for key, state in list(entries.items()):
                if (state.chat_id, state.menu_id) == (chat_id, menu_id):
                    del entries[key]
        self.processing.intersection_update(self.inputs)
