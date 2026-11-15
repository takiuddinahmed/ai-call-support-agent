import pytest

from support_agent.config import Settings
from support_agent.store.client import StoreClient


@pytest.fixture
async def store():
    client = StoreClient(Settings(_env_file=None))
    yield client
    await client.aclose()
