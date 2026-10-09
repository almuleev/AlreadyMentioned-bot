"""Bounded draft/input state; only confirmed publications are persistent."""

import secrets
from collections import OrderedDict
from dataclasses import dataclass, field
from time import monotonic

from already_mentioned.services.faq_data import FaqReview


@dataclass
class FaqDraft:
    token: str
    chat_id: int
    user_id: int
    menu_id: int
    thread_id: int | None
    expires_at: float
    question: str = ""
    answer: str = ""
    review: FaqReview = field(default_factory=lambda: FaqReview(()))
    prompt_id: int | None = None
    kind: str = ""


class FaqSessions:
    def __init__(self, *, clock=monotonic, ttl=900, capacity=50):
        self.clock, self.ttl, self.capacity = clock, ttl, capacity
        self.drafts = OrderedDict()

    def clean(self):
        for token, draft in list(self.drafts.items()):
            if draft.expires_at <= self.clock():
                del self.drafts[token]

    def discard_menu(self, chat_id, menu_id):
        for token, draft in list(self.drafts.items()):
            if (draft.chat_id, draft.menu_id) == (chat_id, menu_id):
                del self.drafts[token]

    def create(self, chat_id, user_id, menu_id, thread_id):
        self.clean()
        self.discard_menu(chat_id, menu_id)
        token = secrets.token_urlsafe(8)
        draft = FaqDraft(
            token, chat_id, user_id, menu_id, thread_id, self.clock() + self.ttl
        )
        self.drafts[token] = draft
        while len(self.drafts) > self.capacity:
            self.drafts.popitem(last=False)
        return draft

    def get(self, token, chat_id, user_id, menu_id):
        self.clean()
        draft = self.drafts.get(token)
        return (
            draft
            if draft
            and (draft.chat_id, draft.user_id, draft.menu_id)
            == (chat_id, user_id, menu_id)
            else None
        )

    def input(self, chat_id, prompt_id):
        self.clean()
        return next(
            (
                d
                for d in self.drafts.values()
                if (d.chat_id, d.prompt_id) == (chat_id, prompt_id)
            ),
            None,
        )

    def remove(self, token):
        self.drafts.pop(token, None)
