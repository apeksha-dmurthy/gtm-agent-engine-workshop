import copy
import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile, score_prospect
from gtm_agent.gtm_records import OFFERINGS, PROSPECTS


class UpdateProspectInfoTests(unittest.TestCase):
    prospect_id = "LEAD-71001"

    def setUp(self):
        self.record_patch = patch.dict(
            PROSPECTS,
            {self.prospect_id: copy.deepcopy(PROSPECTS[self.prospect_id])},
        )
        self.record_patch.start()
        self.profile_patch = patch.dict(
            data_service._PROFILES,
            {self.prospect_id: {"prospect_id": self.prospect_id, "tech_stack": []}},
        )
        self.profile_patch.start()

    def tearDown(self):
        self.profile_patch.stop()
        self.record_patch.stop()

    def test_update_persists_technology_and_refreshes_cached_profile(self):
        self.assertNotIn("Kafka", data_service.fetch_tech_stack(self.prospect_id))

        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertEqual(set(result), {"updated", "found", "tech_stack"})
        self.assertEqual(result["tech_stack"], data_service.fetch_tech_stack(self.prospect_id))
        self.assertIn("Kafka", data_service.fetch_tech_stack(self.prospect_id))
        self.assertIsNone(data_service.get_profile_from_db(self.prospect_id)["prospect_profile"])

        refreshed = build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertIn("Kafka", refreshed["prospect_profile"]["tech_stack"])

    def test_scoring_uses_technology_added_after_profile_was_cached(self):
        cached_profile = build_prospect_profile.invoke({"prospect_id": self.prospect_id})[
            "prospect_profile"
        ]
        data_service.update_prospect_info(self.prospect_id, "Kafka")
        fake_result = Mock()
        fake_result.model_dump.return_value = {"score": 100}
        fake_llm = Mock()
        fake_llm.invoke.return_value = fake_result

        with patch("gtm_agent.gtm_agent._scoring_llm", fake_llm):
            score_prospect.invoke(
                {
                    "prospect_profile": cached_profile,
                    "offering": OFFERINGS["OFFER-10001"],
                }
            )

        scoring_input = json.loads(
            fake_llm.invoke.call_args.args[0][1]["content"].split(
                "\n\nProspect profile:\n", 1
            )[1]
        )
        self.assertIn("Kafka", scoring_input["tech_stack"])
        missing_technologies = set(OFFERINGS["OFFER-10001"]["required_tech_stack"]) - set(
            scoring_input["tech_stack"]
        )
        self.assertNotIn("Kafka", missing_technologies)


if __name__ == "__main__":
    unittest.main()
