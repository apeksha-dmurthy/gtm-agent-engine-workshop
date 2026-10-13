import os
import unittest
from types import SimpleNamespace

os.environ.setdefault("OPENAI_API_KEY", "test")

from gtm_agent import data_service
from gtm_agent.gtm_agent import SYSTEM_PROMPT, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def send(self, prospect):
        return send_prospect_email.func(
            prospect=prospect,
            subject="Hello",
            body="Checking in.",
            runtime=SimpleNamespace(config={}),
            from_rep={},
        )

    def test_disqualified_lead_is_blocked(self):
        prospect = {
            "prospect_id": "LEAD-50001",
            "email": "priya.nair@brightwaveapps.com",
        }

        result = self.send(prospect)

        self.assertEqual(result, {
            "status": "blocked",
            "reason": "Prospect is disqualified.",
        })

    def test_eligible_prospect_is_sent(self):
        prospect_id, record = next(
            (prospect_id, record)
            for prospect_id, record in data_service.PROSPECTS.items()
            if not record.get("disqualified")
        )

        result = self.send({"prospect_id": prospect_id, **record})

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], record["email"])

    def test_disqualified_email_fallback_is_blocked(self):
        prospect = {
            "prospect_id": "UNKNOWN-LEAD",
            "email": "priya.nair@brightwaveapps.com",
        }

        result = self.send(prospect)

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["reason"], "Prospect is disqualified.")

    def test_missing_email_remains_failed(self):
        result = self.send({"prospect_id": "LEAD-50001"})

        self.assertEqual(result, {
            "status": "failed",
            "error": "Prospect record has no email address.",
        })

    def test_prompt_instructs_agent_to_report_blocked_send(self):
        self.assertIn("explain the tool's reason to the rep", SYSTEM_PROMPT)
        self.assertIn("do not say the email was sent", SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
