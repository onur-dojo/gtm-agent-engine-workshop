import os
import unittest

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class EmailInstructionTests(unittest.TestCase):
    def test_document_request_requires_link_before_sending(self):
        prompt = SYSTEM_PROMPT.lower()

        self.assertIn("document or file", prompt)
        self.assertIn("attachments are unsupported", prompt)
        self.assertIn("ask for a link", prompt)
        self.assertIn("do not call send_prospect_email until the rep provides a link", prompt)
        self.assertIn("include it in the body and send", prompt)
        self.assertIn("for other email requests, always carry the request through", prompt)

    def test_email_tool_documents_text_only_limitation(self):
        self.assertIn("text-only", send_prospect_email.description)
        self.assertIn("attachments are not supported", send_prospect_email.description)
        self.assertNotIn("attachment", send_prospect_email.args_schema.model_fields)

    def test_plain_text_email_is_still_sent(self):
        result = send_prospect_email.func(
            prospect={"name": "Prospect", "email": "prospect@example.com"},
            subject="Introduction",
            body="Would you be available for a call?",
            runtime=None,
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], "Would you be available for a call?")


if __name__ == "__main__":
    unittest.main()
