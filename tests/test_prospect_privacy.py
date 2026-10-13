import os
import unittest
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("LANGSMITH_TRACING", "false")

from gtm_agent import gtm_agent


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.prospect = {
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "annual_revenue": 5000000,
            "enrichment_source": "CRM",
            "disqualified": False,
            "billing_qualification": {"tax_id": "sensitive-tax-id"},
        }

    def test_get_prospect_returns_only_allowed_contact_fields(self):
        with patch.object(gtm_agent.data_service, "get_prospect_record", return_value=self.prospect):
            result = gtm_agent.get_prospect.invoke({"prospect_id": "LEAD-1"})

        self.assertEqual(result["prospect"], {
            "name": "Ada Lovelace",
            "email": "ada@example.com",
            "annual_revenue": 5000000,
            "disqualified": False,
        })
        self.assertNotIn("billing_qualification", result["prospect"])

    def test_build_prospect_profile_uses_allowed_source_fields(self):
        with (
            patch.object(gtm_agent.data_service, "get_profile_from_db", return_value={"prospect_profile": None}),
            patch.object(gtm_agent.data_service, "get_prospect_record", return_value=self.prospect),
            patch.object(gtm_agent.data_service, "fetch_engagement_history", return_value=[]),
            patch.object(gtm_agent.data_service, "fetch_account_details", return_value={}),
            patch.object(gtm_agent.data_service, "fetch_tech_stack", return_value=[]),
            patch.object(gtm_agent.data_service, "save_profile_to_db"),
        ):
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        profile = result["prospect_profile"]
        self.assertEqual(profile["name"], "Ada Lovelace")
        self.assertEqual(profile["email"], "ada@example.com")
        self.assertEqual(profile["annual_revenue"], 5000000)
        self.assertEqual(profile["enrichment_source"], "CRM")
        self.assertIs(profile["disqualified"], False)
        self.assertNotIn("billing_qualification", profile)

    def test_cached_profile_omits_billing_qualification(self):
        with patch.object(
            gtm_agent.data_service,
            "get_profile_from_db",
            return_value={"prospect_profile": {**self.prospect, "tech_stack": []}},
        ):
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": "LEAD-1"})

        self.assertNotIn("billing_qualification", result["prospect_profile"])

    def test_profile_store_omits_billing_qualification(self):
        with patch.dict(gtm_agent.data_service._PROFILES, {"LEAD-1": self.prospect}):
            result = gtm_agent.data_service.get_profile_from_db("LEAD-1")

        self.assertNotIn("billing_qualification", result["prospect_profile"])

    def test_score_prospect_excludes_billing_from_scoring_prompt(self):
        captured_messages = []

        class ScoringModel:
            def invoke(self, messages):
                captured_messages.extend(messages)
                return type("Score", (), {"model_dump": lambda self: {"score": 80}})()

        offering = {
            "required_tech_stack": ["Python"],
            "min_annual_revenue": 1000000,
            "description": "Data platform",
        }
        profile = {
            "name": "Ada Lovelace",
            "billing_qualification": {"tax_id": "sensitive-tax-id"},
        }
        with patch.object(gtm_agent, "_scoring_llm", ScoringModel()):
            gtm_agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})

        serialized_prompt = " ".join(message["content"] for message in captured_messages)
        self.assertNotIn("billing_qualification", serialized_prompt)
        self.assertNotIn("sensitive-tax-id", serialized_prompt)


if __name__ == "__main__":
    unittest.main()
