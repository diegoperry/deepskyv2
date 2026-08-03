from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.web_app import (
    AuthUser,
    _apply_subscription_update,
    _apply_credit_purchase,
    _consume_credit_or_require_subscription,
    create_credit_checkout,
    _is_paid_profile,
    _html,
    _reconcile_stripe_subscription,
    _subscription_matches_price,
)


PRICE_ID = "price_deepsky"


def _subscription(*, status: str = "active", period_offset: int = 3600, price_id: str = PRICE_ID) -> dict:
    return {
        "id": "sub_test",
        "customer": "cus_test",
        "status": status,
        "current_period_end": int(time.time()) + period_offset,
        "metadata": {"user_id": "user_test"},
        "items": {"data": [{"price": {"id": price_id}}]},
    }


class BillingEntitlementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.user = AuthUser(id="user_test", email="test@example.com")

    def test_credit_pack_prices_are_visible_in_processing_ui(self) -> None:
        html = _html()
        for marker in (
            'id="buyCredits"',
            'data-credit-pack="5"',
            'data-credit-pack="10"',
            'data-credit-pack="20"',
            '/api/billing/credits/checkout',
        ):
            self.assertIn(marker, html)
    def test_only_active_or_trialing_with_future_period_is_paid(self) -> None:
        future = "2999-01-01T00:00:00Z"
        past = "2000-01-01T00:00:00Z"
        self.assertTrue(_is_paid_profile({"subscription_status": "active", "current_period_end": future}))
        self.assertTrue(_is_paid_profile({"subscription_status": "trialing", "current_period_end": future}))
        for status in ("canceled", "past_due", "unpaid", "paused", "incomplete", "free"):
            self.assertFalse(_is_paid_profile({"subscription_status": status, "current_period_end": future}))
        self.assertFalse(_is_paid_profile({"subscription_status": "active", "current_period_end": past}))
        self.assertFalse(_is_paid_profile({"subscription_status": "active", "current_period_end": None}))

    @patch("app.web_app._stripe_checkout_price_id", return_value="price_new_20")
    @patch("app.web_app._stripe_price_id", return_value=PRICE_ID)
    def test_legacy_and_new_subscription_prices_are_both_accepted(self, _legacy, _new) -> None:
        self.assertTrue(_subscription_matches_price(_subscription(price_id=PRICE_ID)))
        self.assertTrue(_subscription_matches_price(_subscription(price_id="price_new_20")))
        self.assertFalse(_subscription_matches_price(_subscription(price_id="price_other")))
    @patch("app.web_app._stripe_price_id", return_value=PRICE_ID)
    @patch("app.web_app._update_profile")
    def test_wrong_price_webhook_revokes_paid_status(self, update_profile, _price_id) -> None:
        _apply_subscription_update(_subscription(price_id="price_other"))
        self.assertEqual(update_profile.call_args.args[1]["subscription_status"], "canceled")

    @patch("app.web_app._stripe_price_id", return_value=PRICE_ID)
    @patch("app.web_app._update_profile")
    def test_expected_price_webhook_keeps_active_status(self, update_profile, _price_id) -> None:
        _apply_subscription_update(_subscription())
        self.assertEqual(update_profile.call_args.args[1]["subscription_status"], "active")

    @patch("app.web_app._update_profile")
    def test_deleted_subscription_revokes_access_even_before_period_end(self, update_profile) -> None:
        _apply_subscription_update(_subscription(), status_override="canceled")
        updates = update_profile.call_args.args[1]
        self.assertEqual(updates["subscription_status"], "canceled")
        self.assertFalse(_is_paid_profile(updates))

    @patch("app.web_app._get_profile")
    @patch("app.web_app._update_profile")
    @patch("app.web_app._stripe_module")
    @patch("app.web_app._stripe_price_id", return_value=PRICE_ID)
    def test_reconciliation_revokes_when_no_matching_subscription(
        self, _price_id, stripe_module, update_profile, get_profile
    ) -> None:
        stripe_module.return_value.Subscription.list.return_value = {"data": [_subscription(price_id="price_other")]}
        get_profile.return_value = None
        profile = {
            "user_id": self.user.id,
            "stripe_customer_id": "cus_test",
            "subscription_status": "active",
            "current_period_end": "2000-01-01T00:00:00Z",
        }
        reconciled = _reconcile_stripe_subscription(self.user, profile)
        self.assertFalse(_is_paid_profile(reconciled))
        self.assertEqual(update_profile.call_args.args[1]["subscription_status"], "canceled")

    @patch("app.web_app._reconcile_stripe_subscription")
    @patch("app.web_app._billing_profile_for")
    def test_paid_user_does_not_consume_credit(self, billing_profile, reconcile) -> None:
        profile = {"subscription_status": "active", "current_period_end": "2999-01-01T00:00:00Z"}
        billing_profile.return_value = profile
        reconcile.return_value = profile
        returned, consumed = _consume_credit_or_require_subscription(self.user)
        self.assertIs(returned, profile)
        self.assertFalse(consumed)

    @patch("app.web_app._supabase_rest_request")
    @patch("app.web_app._reconcile_stripe_subscription")
    @patch("app.web_app._billing_profile_for")
    def test_canceled_user_without_credits_is_denied(self, billing_profile, reconcile, rest_request) -> None:
        profile = {
            "subscription_status": "canceled",
            "current_period_end": "2999-01-01T00:00:00Z",
            "free_credits_remaining": 0,
        }
        billing_profile.return_value = profile
        reconcile.return_value = profile
        with self.assertRaises(HTTPException) as raised:
            _consume_credit_or_require_subscription(self.user)
        self.assertEqual(raised.exception.status_code, 402)
        rest_request.assert_not_called()

    @patch("app.web_app._get_profile", return_value={"free_credits_remaining": 1})
    @patch("app.web_app._supabase_rest_request", return_value=True)
    @patch("app.web_app._reconcile_stripe_subscription")
    @patch("app.web_app._billing_profile_for")
    def test_non_paid_user_consumes_one_atomic_credit(self, billing_profile, reconcile, rest_request, _get_profile) -> None:
        profile = {"subscription_status": "free", "free_credits_remaining": 2}
        billing_profile.return_value = profile
        reconcile.return_value = profile
        _, consumed = _consume_credit_or_require_subscription(self.user)
        self.assertTrue(consumed)
        rest_request.assert_called_once_with(
            "rpc/consume_free_credit", method="POST", payload={"target_user_id": self.user.id}
        )

    @patch("app.web_app._credit_pack_price_id", return_value="price_pack_10")
    @patch("app.web_app._stripe_module")
    @patch("app.web_app._billing_profile_for", return_value={"stripe_customer_id": "cus_test"})
    def test_credit_checkout_is_one_time_and_uses_selected_price(self, _profile, stripe_module, _price) -> None:
        stripe_module.return_value.checkout.Session.create.return_value.url = "https://checkout.test/session"
        result = create_credit_checkout({"pack": "10"}, self.user)
        self.assertEqual(result["url"], "https://checkout.test/session")
        kwargs = stripe_module.return_value.checkout.Session.create.call_args.kwargs
        self.assertEqual(kwargs["mode"], "payment")
        self.assertEqual(kwargs["line_items"], [{"price": "price_pack_10", "quantity": 1}])
        self.assertEqual(kwargs["metadata"]["pack"], "10")
    @patch("app.web_app._supabase_rest_request", return_value=True)
    def test_paid_credit_checkout_grants_fixed_pack_atomically(self, rest_request) -> None:
        applied = _apply_credit_purchase({
            "id": "cs_test",
            "payment_status": "paid",
            "client_reference_id": self.user.id,
            "metadata": {"purchase_type": "image_credits", "pack": "10"},
        })
        self.assertTrue(applied)
        rest_request.assert_called_once_with(
            "rpc/grant_image_credit_purchase",
            method="POST",
            payload={"target_user_id": self.user.id, "checkout_session_id": "cs_test", "credits_to_add": 10},
        )

    @patch("app.web_app._supabase_rest_request")
    def test_unpaid_credit_checkout_does_not_grant_credits(self, rest_request) -> None:
        applied = _apply_credit_purchase({
            "id": "cs_unpaid",
            "payment_status": "unpaid",
            "client_reference_id": self.user.id,
            "metadata": {"purchase_type": "image_credits", "pack": "20"},
        })
        self.assertFalse(applied)
        rest_request.assert_not_called()

if __name__ == "__main__":
    unittest.main()
