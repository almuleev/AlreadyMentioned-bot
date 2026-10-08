"""Bound duplicate join/deep-link greetings without storing wizard state."""

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from time import monotonic


class WelcomeMessages:
    def __init__(self, *, clock=monotonic, ttl=60.0, capacity=1000):
        self.clock = clock
        self.ttl = ttl
        self.capacity = capacity
        self.sent: OrderedDict[int, float] = OrderedDict()
        self.lock = asyncio.Lock()

    async def send_once(self, chat_id: int, send: Callable[[], Awaitable[bool | None]]):
        async with self.lock:
            now = self.clock()
            while self.sent and next(iter(self.sent.values())) <= now - self.ttl:
                self.sent.popitem(last=False)
            if chat_id in self.sent:
                return
            if await send() is False:
                return
            self.sent[chat_id] = self.clock()
            while len(self.sent) > self.capacity:
                self.sent.popitem(last=False)
