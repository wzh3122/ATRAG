import asyncio
import unittest

from atrag.agent.hybrid_router import (
    HybridAgentRouter,
    LLMHybridAgentRouter,
    RouteMode,
    RouterLLMConfig,
)


class HybridAgentRouterTest(unittest.TestCase):
    def setUp(self):
        self.router = HybridAgentRouter()

    def test_simple_question_routes_direct_without_tools(self):
        decision = self.router.route(
            "Explain recursion in one sentence",
            has_collections=False,
            collections_explicit=False,
            has_chat_files=False,
            web_search_enabled=False,
        )

        self.assertEqual(decision.mode, RouteMode.DIRECT)
        self.assertEqual(decision.candidate_tools, ())

    def test_explicit_collection_routes_to_collection_search(self):
        decision = self.router.route(
            "Summarize the deployment design",
            has_collections=True,
            collections_explicit=True,
            has_chat_files=False,
            web_search_enabled=False,
        )

        self.assertEqual(decision.mode, RouteMode.KNOWLEDGE)
        self.assertEqual(decision.candidate_tools, ("search_collection",))

    def test_retrieval_without_selected_collection_discovers_collections(self):
        decision = self.router.route(
            "Search for the quota design and cite sources",
            has_collections=False,
            collections_explicit=False,
            has_chat_files=False,
            web_search_enabled=False,
        )

        self.assertEqual(decision.mode, RouteMode.KNOWLEDGE)
        self.assertEqual(decision.candidate_tools, ("list_collections", "search_collection"))

    def test_attached_file_beats_default_collection(self):
        decision = self.router.route(
            "请总结一下",
            has_collections=True,
            collections_explicit=False,
            has_chat_files=True,
            web_search_enabled=False,
        )

        self.assertEqual(decision.mode, RouteMode.CHAT_FILES)
        self.assertEqual(decision.candidate_tools, ("search_chat_files",))

    def test_collection_and_enabled_web_intent_routes_hybrid(self):
        decision = self.router.route(
            "结合知识库和最新新闻分析这个变化",
            has_collections=True,
            collections_explicit=True,
            has_chat_files=False,
            web_search_enabled=True,
        )

        self.assertEqual(decision.mode, RouteMode.HYBRID)
        self.assertEqual(
            decision.candidate_tools, ("search_collection", "web_search", "web_read")
        )

    def test_explicit_web_intent_beats_unselected_default_collection(self):
        decision = self.router.route(
            "What is the latest weather today?",
            has_collections=True,
            collections_explicit=False,
            has_chat_files=False,
            web_search_enabled=True,
        )

        self.assertEqual(decision.mode, RouteMode.WEB)
        self.assertEqual(decision.candidate_tools, ("web_search", "web_read"))

    def test_disabled_web_search_is_never_a_candidate(self):
        decision = self.router.route(
            "联网查一下今天的新闻",
            has_collections=False,
            collections_explicit=False,
            has_chat_files=False,
            web_search_enabled=False,
        )

        self.assertEqual(decision.mode, RouteMode.DIRECT)
        self.assertNotIn("web_search", decision.candidate_tools)
        self.assertIn("web intent present but web search disabled", decision.signals)

    def test_prompt_context_keeps_original_agent_as_final_decider(self):
        context = self.router.route(
            "search online for the latest release",
            has_collections=False,
            collections_explicit=False,
            has_chat_files=False,
            web_search_enabled=True,
        ).as_prompt_context()

        self.assertIn("second-stage agent", context)
        self.assertIn("You may override", context)
        self.assertIn("`web_search`", context)


class LLMHybridAgentRouterTest(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def config(timeout_seconds=8):
        return RouterLLMConfig(
            provider_name="user-openai",
            custom_llm_provider="openai",
            model="router-model",
            base_url="https://example.test/v1",
            api_key="secret",
            timeout_seconds=timeout_seconds,
        )

    async def test_valid_llm_decision_is_used(self):
        async def caller(config, prompt):
            self.assertEqual(config.model, "router-model")
            self.assertIn('"web_search_enabled": true', prompt)
            return (
                '{"mode":"web","candidate_tools":["web_search","web_read"],'
                '"confidence":0.91,"reason":"fresh information requested"}'
            )

        decision = await LLMHybridAgentRouter(llm_caller=caller).route(
            "查询最新版本",
            has_collections=False,
            collections_explicit=False,
            has_chat_files=False,
            web_search_enabled=True,
            llm_config=self.config(),
        )

        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.mode, RouteMode.WEB)
        self.assertEqual(decision.candidate_tools, ("web_search", "web_read"))

    async def test_llm_exception_uses_rule_fallback(self):
        async def caller(config, prompt):
            raise ConnectionError("provider unavailable")

        decision = await LLMHybridAgentRouter(llm_caller=caller).route(
            "搜索知识库中的部署方案",
            has_collections=True,
            collections_explicit=True,
            has_chat_files=False,
            web_search_enabled=False,
            llm_config=self.config(),
        )

        self.assertEqual(decision.source, "rules_fallback")
        self.assertEqual(decision.mode, RouteMode.KNOWLEDGE)
        self.assertIn("routing LLM failed: ConnectionError", decision.signals)

    async def test_fenced_json_is_accepted_and_reason_is_single_line(self):
        async def caller(config, prompt):
            return (
                '```json\n{"mode":"direct","candidate_tools":[],"confidence":0.8,'
                '"reason":"simple\\nrequest"}\n```'
            )

        decision = await LLMHybridAgentRouter(llm_caller=caller).route(
            "hello",
            has_collections=False,
            collections_explicit=False,
            has_chat_files=False,
            web_search_enabled=False,
            llm_config=self.config(),
        )

        self.assertEqual(decision.source, "llm")
        self.assertEqual(decision.signals, ("routing LLM: simple request",))

    async def test_llm_timeout_uses_rule_fallback(self):
        async def caller(config, prompt):
            await asyncio.sleep(0.05)
            return "{}"

        decision = await LLMHybridAgentRouter(llm_caller=caller).route(
            "Explain recursion",
            has_collections=False,
            collections_explicit=False,
            has_chat_files=False,
            web_search_enabled=False,
            llm_config=self.config(timeout_seconds=0.001),
        )

        self.assertEqual(decision.source, "rules_fallback")
        self.assertEqual(decision.mode, RouteMode.DIRECT)
        self.assertTrue(any("TimeoutError" in signal for signal in decision.signals))

    async def test_unavailable_tool_from_llm_uses_rule_fallback(self):
        async def caller(config, prompt):
            return (
                '{"mode":"web","candidate_tools":["web_search"],'
                '"confidence":0.9,"reason":"requested web"}'
            )

        decision = await LLMHybridAgentRouter(llm_caller=caller).route(
            "联网查询",
            has_collections=False,
            collections_explicit=False,
            has_chat_files=False,
            web_search_enabled=False,
            llm_config=self.config(),
        )

        self.assertEqual(decision.source, "rules_fallback")
        self.assertNotIn("web_search", decision.candidate_tools)

    async def test_missing_llm_configuration_uses_rule_fallback(self):
        decision = await LLMHybridAgentRouter().route(
            "总结附件",
            has_collections=False,
            collections_explicit=False,
            has_chat_files=True,
            web_search_enabled=False,
            llm_config=None,
        )

        self.assertEqual(decision.source, "rules_fallback")
        self.assertEqual(decision.mode, RouteMode.CHAT_FILES)


if __name__ == "__main__":
    unittest.main()
