import asyncio
import unittest
from unittest.mock import patch

from atrag.agent import agent_session_manager
from atrag.agent.agent_config import AgentConfig


def _config() -> AgentConfig:
    return AgentConfig(
        user_id="user-1",
        chat_id="chat-1",
        provider_name="provider-1",
        api_key="secret",
        base_url="https://example.test/v1",
        default_model="model-1",
    )


class AgentSessionManagerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await agent_session_manager.shutdown_all()
        agent_session_manager._session_locks.clear()

    async def asyncTearDown(self):
        await agent_session_manager.shutdown_all()
        agent_session_manager._session_locks.clear()

    async def test_concurrent_requests_share_one_initialized_session(self):
        initialization_count = 0

        async def initialize(session):
            nonlocal initialization_count
            initialization_count += 1
            await asyncio.sleep(0.01)
            session._ready = True

        with patch.object(agent_session_manager.ChatSession, "initialize", new=initialize):
            sessions = await asyncio.gather(
                *(agent_session_manager.get_or_create_session(_config()) for _ in range(10))
            )

        self.assertEqual(initialization_count, 1)
        self.assertTrue(all(session is sessions[0] for session in sessions))
        self.assertEqual(agent_session_manager.get_stats()["total_sessions"], 1)

    async def test_expired_session_is_replaced_once_under_concurrency(self):
        initialization_count = 0

        async def initialize(session):
            nonlocal initialization_count
            initialization_count += 1
            await asyncio.sleep(0.01)
            session._ready = True

        config = _config()
        expired_session = agent_session_manager.ChatSession(config)
        expired_session._ready = True
        expired_session.last_used -= 1801
        agent_session_manager._chat_sessions[config.get_session_key()] = expired_session

        with (
            patch.object(agent_session_manager.ChatSession, "initialize", new=initialize),
            patch.object(expired_session, "_cleanup", return_value=None),
        ):
            sessions = await asyncio.gather(
                *(agent_session_manager.get_or_create_session(config) for _ in range(10))
            )

        self.assertEqual(initialization_count, 1)
        self.assertTrue(all(session is sessions[0] for session in sessions))
        self.assertIsNot(sessions[0], expired_session)


if __name__ == "__main__":
    unittest.main()
