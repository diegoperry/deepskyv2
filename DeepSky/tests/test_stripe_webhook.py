from __future__ import annotations

import unittest
from unittest.mock import patch

from app.web_app import _apply_subscription_update


class _StripeMetadataWithoutGet:
    pass


class _StripeSubscription:
    id = "sub_test"
    customer = "cus_test"
    status = "active"
    metadata = _StripeMetadataWithoutGet()

    def to_dict_recursive(self) -> dict[str, object]:
        return {"metadata": {"user_id": "user_test"}}


class StripeWebhookTests(unittest.TestCase):
    @patch("app.web_app._update_profile")
    def test_subscription_metadata_is_converted_before_user_id_lookup(self, update_profile) -> None:
        _apply_subscription_update(_StripeSubscription())

        update_profile.assert_called_once()
        user_id, updates = update_profile.call_args.args
        self.assertEqual(user_id, "user_test")
        self.assertEqual(updates["stripe_subscription_id"], "sub_test")
        self.assertEqual(updates["stripe_customer_id"], "cus_test")
        self.assertEqual(updates["subscription_status"], "active")


if __name__ == "__main__":
    unittest.main()
