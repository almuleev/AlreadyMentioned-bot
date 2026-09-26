"""Bot creation separated from the long-polling entry point."""

from urllib.request import getproxies

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession

from already_mentioned.config import Settings


def create_bot(settings: Settings) -> Bot:
    proxy = getproxies().get("https")
    session = AiohttpSession(proxy=proxy) if proxy else AiohttpSession()
    return Bot(token=settings.require_bot_token(), session=session)
