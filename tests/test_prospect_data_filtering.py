import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gtm_agent import data_service
from gtm_agent import gtm_agent


class ProspectDataFilteringTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-TEST-1"
        self.record = {
            "name": "Test Prospect",
            "email": "test@example.com",
            "billing_qualification": {
                "tax_id": "test-tax-id",
                "date_of_birth": "test-date-of-birth",
                "card_on_file": "test-card",
            },
            "disqualified": False,
            "annual_revenue": 42000000,
            "enrichment_source": "test-source",
            "engagement_history": [],
            "account_details": [],
            "tech_stack": ["Snowflake"],
        }

    def assert_no_billing_fields(self, value):
        serialized = json.dumps(value)
        for field in ("tax_id", "date_of_birth", "card_on_file"):
            self.assertNotIn(field, serialized)

    def test_prospect_record_and_contact_tool_exclude_billing_data(self):
        with patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record}):
            record = data_service.get_prospect_record(self.prospect_id)
            result = gtm_agent.get_prospect.invoke({"prospect_id": self.prospect_id})

        self.assertNotIn("billing_qualification", record)
        self.assert_no_billing_fields(record)
        self.assert_no_billing_fields(result)
        self.assertEqual(
            result["prospect"],
            {
                "prospect_id": self.prospect_id,
                "name": "Test Prospect",
                "email": "test@example.com",
                "disqualified": False,
                "annual_revenue": 42000000,
                "enrichment_source": "test-source",
            },
        )

    def test_built_and_cached_profiles_exclude_billing_data(self):
        with (
            patch.dict(data_service.PROSPECTS, {self.prospect_id: self.record}),
            patch.dict(data_service._PROFILES, {}, clear=True),
        ):
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            profile = result["prospect_profile"]
            cached = data_service.get_profile_from_db(self.prospect_id)["prospect_profile"]

            self.assertNotIn("billing_qualification", profile)
            self.assertNotIn("billing_qualification", cached)
            self.assert_no_billing_fields(profile)
            self.assert_no_billing_fields(cached)
            self.assertEqual(profile["annual_revenue"], 42000000)
            self.assertEqual(profile["tech_stack"], ["Snowflake"])

        stale_profile = {**self.record, "prospect_id": self.prospect_id}
        with patch.dict(data_service._PROFILES, {self.prospect_id: stale_profile}, clear=True):
            result = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
            self.assertNotIn("billing_qualification", result["prospect_profile"])
            self.assert_no_billing_fields(result)
            self.assertNotIn("billing_qualification", data_service._PROFILES[self.prospect_id])

    def test_scorer_prompt_excludes_billing_and_keeps_scoring_fields(self):
        profile = {
            "prospect_id": self.prospect_id,
            "annual_revenue": 42000000,
            "billing_qualification": self.record["billing_qualification"],
            "tech_stack": [],
        }
        offering = {
            "required_tech_stack": ["Snowflake"],
            "min_annual_revenue": 1000000,
            "description": "Test offering",
        }
        scorer = SimpleNamespace(model_dump=lambda: {"score": 75})
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["Snowflake"]),
            patch.object(gtm_agent, "_scoring_llm") as scoring_llm,
        ):
            scoring_llm.invoke.return_value = scorer
            result = gtm_agent.score_prospect.invoke(
                {"prospect_profile": profile, "offering": offering}
            )

        prompt = json.dumps(scoring_llm.invoke.call_args.args[0])
        for field in ("tax_id", "date_of_birth", "card_on_file"):
            self.assertNotIn(field, prompt)
        self.assertIn("annual_revenue", prompt)
        self.assertIn("Snowflake", prompt)
        self.assertEqual(result["score"], 75)


if __name__ == "__main__":
    unittest.main()
