import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import data_service
from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {}})

    def test_disqualified_prospect_is_blocked(self):
        prospect = {
            "prospect_id": "LEAD-50001",
            **data_service.get_prospect_record("LEAD-50001"),
        }

        with patch("gtm_agent.gtm_agent.uuid.uuid4") as uuid4:
            result = send_prospect_email.func(
                prospect, "Intro", "Hello", self.runtime
            )

        self.assertEqual(
            result,
            {"status": "blocked", "error": "Prospect is disqualified; email not sent."},
        )
        uuid4.assert_not_called()

    def test_eligible_prospect_is_sent(self):
        prospect = {
            "prospect_id": "LEAD-15229",
            **data_service.get_prospect_record("LEAD-15229"),
        }

        result = send_prospect_email.func(prospect, "Intro", "Hello", self.runtime)

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], prospect["email"])

    def test_prompt_tells_rep_when_prospect_is_disqualified(self):
        self.assertIn(
            "do not email prospects whose record is marked disqualified; instead tell the rep the prospect is disqualified",
            SYSTEM_PROMPT,
        )


if __name__ == "__main__":
    unittest.main()
