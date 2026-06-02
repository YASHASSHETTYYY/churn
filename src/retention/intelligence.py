from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.models.predict import ChurnPredictor


DEFAULT_MONTHLY_REVENUE = 65.0
DEFAULT_DISCOUNT_PERCENT = 10.0


def format_feature_name(column: str) -> str:
    return column.replace("_", " ").title()


def risk_tier(probability: float) -> str:
    if probability < 0.2:
        return "Low"
    if probability <= 0.5:
        return "Medium"
    return "High"


def _as_number(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _set_non_negative(customer: dict[str, Any], field: str, value: float) -> None:
    if field in customer:
        customer[field] = max(0.0, float(value))


def _scale_charge_pair(
    customer: dict[str, Any],
    minutes_field: str,
    charge_field: str,
    percent_delta: float,
) -> None:
    if minutes_field not in customer and charge_field not in customer:
        return
    factor = max(0.0, 1.0 + percent_delta / 100.0)
    _set_non_negative(customer, minutes_field, _as_number(customer.get(minutes_field)) * factor)
    _set_non_negative(customer, charge_field, _as_number(customer.get(charge_field)) * factor)


def _driver_theme(feature_name: str) -> str:
    lowered = feature_name.lower()
    if "customer_service" in lowered:
        return "service"
    if "international" in lowered or "intl" in lowered:
        return "international"
    if "charge" in lowered or "minutes" in lowered or "plan" in lowered:
        return "plan_fit"
    if "vmail" in lowered or "voice_mail" in lowered:
        return "engagement"
    if "account_length" in lowered:
        return "tenure"
    return "general"


def _reason_for_driver(feature: str, value: Any, shap_value: float) -> str:
    direction = "raises" if shap_value >= 0 else "lowers"
    readable = format_feature_name(feature)
    theme = _driver_theme(feature)
    if theme == "service":
        detail = "support friction is one of the strongest churn signals"
    elif theme == "international":
        detail = "international usage or plan fit may be creating price pressure"
    elif theme == "plan_fit":
        detail = "usage and charges suggest the current plan may not match behavior"
    elif theme == "engagement":
        detail = "product engagement changes how sticky this account appears"
    elif theme == "tenure":
        detail = "relationship maturity changes switching likelihood"
    else:
        detail = "the model treats this customer signal as meaningful"
    return f"{readable} is {value}, which {direction} churn risk because {detail}."


def _recommendation_for_driver(feature: str, shap_value: float) -> str:
    theme = _driver_theme(feature)
    if theme == "service":
        return "Open a proactive service recovery case, resolve the root issue, and follow up within 48 hours."
    if theme == "international":
        return "Offer an international-plan review with a lower-friction bundle or temporary international credit."
    if theme == "plan_fit":
        return "Recommend a plan fit review and compare the bill against a better-matched package."
    if theme == "engagement":
        return "Use an adoption message or bundle nudge that increases attachment to sticky product features."
    if theme == "tenure":
        return "Use an early-life retention touchpoint with onboarding support and a modest loyalty incentive."
    if shap_value >= 0:
        return "Route the account to a retention specialist with the top SHAP drivers attached."
    return "Preserve the protective behavior and avoid generic discounting unless risk rises."


def _offer_for_driver(feature: str) -> dict[str, Any]:
    theme = _driver_theme(feature)
    if theme == "service":
        return {
            "offer": "Service recovery credit",
            "discount_percent": 8.0,
            "rationale": "A small credit paired with issue resolution targets service-driven churn.",
        }
    if theme == "international":
        return {
            "offer": "International bundle trial",
            "discount_percent": 12.0,
            "rationale": "International-plan pressure is best handled with package fit, not a generic coupon.",
        }
    if theme == "plan_fit":
        return {
            "offer": "Plan optimization discount",
            "discount_percent": 10.0,
            "rationale": "The account appears sensitive to usage-price mismatch.",
        }
    if theme == "engagement":
        return {
            "offer": "Feature adoption bundle",
            "discount_percent": 5.0,
            "rationale": "Engagement-led retention should avoid over-discounting.",
        }
    return {
        "offer": "Targeted save offer",
        "discount_percent": DEFAULT_DISCOUNT_PERCENT,
        "rationale": "Use a controlled incentive because no single operational theme dominates.",
    }


@dataclass
class RetentionIntelligence:
    predictor: ChurnPredictor

    def analyze_customer(
        self,
        customer: dict[str, Any],
        *,
        monthly_revenue: float = DEFAULT_MONTHLY_REVENUE,
        top_k: int = 5,
    ) -> dict[str, Any]:
        prediction = self.predictor.predict_one(customer)
        explanation = self.predictor.explain(customer, top_k=top_k)
        probability = float(prediction["churn_probability"])
        top_factors = explanation.get("top_factors", [])
        risk = risk_tier(probability)

        risk_drivers = [
            factor for factor in top_factors if float(factor.get("shap_value", 0.0)) >= 0
        ]
        primary_factors = risk_drivers or top_factors
        primary = primary_factors[0] if primary_factors else {}
        primary_feature = str(primary.get("feature", "customer profile"))
        primary_value = primary.get("feature_value", "unknown")
        primary_shap = float(primary.get("shap_value", 0.0))

        reasons = [
            _reason_for_driver(
                str(factor["feature"]),
                factor.get("feature_value"),
                float(factor.get("shap_value", 0.0)),
            )
            for factor in primary_factors[:3]
        ]
        recommendations = list(
            dict.fromkeys(
                _recommendation_for_driver(
                    str(factor["feature"]),
                    float(factor.get("shap_value", 0.0)),
                )
                for factor in primary_factors[:3]
            )
        )
        while len(recommendations) < 3:
            recommendations.append(
                "Monitor the account after outreach and rescore before making a larger concession."
            )

        summary = (
            f"This customer is {risk.lower()} risk with a {probability:.1%} churn probability. "
            f"The clearest model signal is {format_feature_name(primary_feature)}={primary_value}, "
            f"which contributes {primary_shap:+.3f} to the churn score."
        )
        offer = _offer_for_driver(primary_feature)
        revenue = self.estimate_revenue_saved(
            baseline_probability=probability,
            improved_probability=max(0.0, probability - self._expected_offer_lift(offer)),
            monthly_revenue=monthly_revenue,
            discount_percent=float(offer["discount_percent"]),
        )

        return {
            "prediction": prediction,
            "explanation": explanation,
            "analyst": {
                "summary": summary,
                "risk_tier": risk,
                "plain_english_reasons": reasons,
                "recommendations": recommendations[:3],
            },
            "agents": {
                "prediction_agent": {
                    "churn_probability": probability,
                    "risk_tier": risk,
                    "label": prediction["churn"],
                },
                "cause_agent": {
                    "top_causes": reasons,
                    "dominant_theme": _driver_theme(primary_feature),
                },
                "offer_agent": offer,
                "revenue_agent": revenue,
            },
        }

    def simulate_digital_twin(
        self,
        customer: dict[str, Any],
        interventions: dict[str, Any],
        *,
        monthly_revenue: float = DEFAULT_MONTHLY_REVENUE,
    ) -> dict[str, Any]:
        baseline_prediction = self.predictor.predict_one(customer)
        twin = self.apply_interventions(customer, interventions)
        twin_prediction = self.predictor.predict_one(twin)

        baseline_probability = float(baseline_prediction["churn_probability"])
        twin_probability = float(twin_prediction["churn_probability"])
        discount_percent = _as_number(interventions.get("discount_percent"), 0.0)
        revenue = self.estimate_revenue_saved(
            baseline_probability=baseline_probability,
            improved_probability=twin_probability,
            monthly_revenue=monthly_revenue,
            discount_percent=discount_percent,
        )
        return {
            "baseline": {
                "customer": customer,
                "prediction": baseline_prediction,
                "risk_tier": risk_tier(baseline_probability),
            },
            "intervention": {
                "inputs": interventions,
                "customer": twin,
                "prediction": twin_prediction,
                "risk_tier": risk_tier(twin_probability),
            },
            "impact": {
                "absolute_probability_change": twin_probability - baseline_probability,
                "relative_probability_change": (
                    (twin_probability - baseline_probability) / baseline_probability
                    if baseline_probability
                    else 0.0
                ),
                "risk_tier_change": f"{risk_tier(baseline_probability)} -> {risk_tier(twin_probability)}",
                **revenue,
            },
        }

    def apply_interventions(
        self,
        customer: dict[str, Any],
        interventions: dict[str, Any],
    ) -> dict[str, Any]:
        twin = dict(customer)
        service_calls_delta = _as_number(interventions.get("service_calls_delta"), 0.0)
        service_calls = _as_number(twin.get("number_customer_service_calls"), 0.0)
        _set_non_negative(
            twin,
            "number_customer_service_calls",
            service_calls + service_calls_delta,
        )

        discount_percent = _as_number(interventions.get("discount_percent"), 0.0)
        if discount_percent:
            charge_factor = max(0.0, 1.0 - discount_percent / 100.0)
            for charge_field in [
                "total_day_charge",
                "total_eve_charge",
                "total_night_charge",
                "total_intl_charge",
            ]:
                _set_non_negative(twin, charge_field, _as_number(twin.get(charge_field)) * charge_factor)

        plan_changes = interventions.get("plan_changes") or {}
        if isinstance(plan_changes, dict):
            for field in ["international_plan", "voice_mail_plan"]:
                if field in plan_changes and field in twin:
                    twin[field] = str(plan_changes[field])

        for period in ["day", "eve", "night", "intl"]:
            percent_key = f"{period}_usage_delta_percent"
            if percent_key in interventions:
                _scale_charge_pair(
                    twin,
                    f"total_{period}_minutes",
                    f"total_{period}_charge",
                    _as_number(interventions[percent_key]),
                )
        return twin

    @staticmethod
    def estimate_revenue_saved(
        *,
        baseline_probability: float,
        improved_probability: float,
        monthly_revenue: float,
        discount_percent: float = 0.0,
        horizon_months: int = 12,
    ) -> dict[str, float]:
        annual_revenue = float(monthly_revenue) * horizon_months
        gross_saved = max(0.0, baseline_probability - improved_probability) * annual_revenue
        discount_cost = max(0.0, float(discount_percent)) / 100.0 * annual_revenue
        return {
            "monthly_revenue": float(monthly_revenue),
            "horizon_months": float(horizon_months),
            "gross_revenue_saved": gross_saved,
            "estimated_offer_cost": discount_cost,
            "net_revenue_saved": gross_saved - discount_cost,
        }

    @staticmethod
    def _expected_offer_lift(offer: dict[str, Any]) -> float:
        discount = _as_number(offer.get("discount_percent"), DEFAULT_DISCOUNT_PERCENT)
        return min(0.18, max(0.03, discount / 100.0))


def build_retention_intelligence(predictor: ChurnPredictor) -> RetentionIntelligence:
    return RetentionIntelligence(predictor=predictor)
