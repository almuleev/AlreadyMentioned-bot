"""Bound, expiring input requests and owner-scoped threshold confirmations."""

import secrets
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic


@dataclass(frozen=True, slots=True)
class SettingsInput:
    chat_id: int
    user_id: int
    menu_id: int
    kind: str
    expires_at: float


@dataclass(frozen=True, slots=True)
class ThresholdPreview:
    chat_id: int
    user_id: int
    menu_id: int
    mode: str
    previous: float | None
    proposed: float
    expires_at: float


class SettingsSessions:
    def __init__(self, *, clock=monotonic, ttl=300, capacity=500):
        self.clock, self.ttl, self.capacity = clock, ttl, capacity
        self.inputs = OrderedDict()
        self.previews = OrderedDict()

    def _clean(self, collection):
        for key, request in list(collection.items()):
            if request.expires_at <= self.clock():
                del collection[key]
        while len(collection) > self.capacity:
            collection.popitem(last=False)

    def add_input(self, chat_id, prompt_id, user_id, menu_id, kind):
        self.inputs[(chat_id, prompt_id)] = SettingsInput(
            chat_id, user_id, menu_id, kind, self.clock() + self.ttl
        )
        self._clean(self.inputs)

    def get_input(self, chat_id, prompt_id):
        self._clean(self.inputs)
        return self.inputs.get((chat_id, prompt_id))

    def has_input(self, chat_id, user_id, menu_id, kind):
        self._clean(self.inputs)
        return any(
            (r.chat_id, r.user_id, r.menu_id, r.kind)
            == (chat_id, user_id, menu_id, kind)
            for r in self.inputs.values()
        )

    def remove_input(self, chat_id, prompt_id):
        self.inputs.pop((chat_id, prompt_id), None)

    def discard_previews(self, chat_id, menu_id):
        for key, request in list(self.previews.items()):
            if (request.chat_id, request.menu_id) == (chat_id, menu_id):
                del self.previews[key]

    def preview(self, chat_id, user_id, menu_id, mode, previous, proposed):
        self.discard_previews(chat_id, menu_id)
        token = secrets.token_urlsafe(8)
        self.previews[token] = ThresholdPreview(
            chat_id, user_id, menu_id, mode, previous, proposed, self.clock() + self.ttl
        )
        self._clean(self.previews)
        return token

    def take_preview(self, token, chat_id, user_id, menu_id):
        self._clean(self.previews)
        request = self.previews.get(token)
        if request and (request.chat_id, request.user_id, request.menu_id) == (
            chat_id,
            user_id,
            menu_id,
        ):
            del self.previews[token]
            return request
        return None
