"""Album bridge pure-logic tests (no network, no GPU)."""
import pytest
from app.agents.orchestrator.bridge import (
    order_tags_genre_first, energy_to_duration_s, build_steering_prose,
)
from app.agents.orchestrator.album import BudgetExceeded
from app.agents.orchestrator import BudgetState


def test_genre_first_ordering():
    tags = order_tags_genre_first(["fingerpicked guitar", "Indie Folk", "tape saturation", "dream pop"])
    assert tags[0] in ("Indie Folk", "dream pop")
    assert len(tags) == 4

def test_ordering_caps_at_six_and_defaults_pop():
    assert len(order_tags_genre_first([f"tag{i}" for i in range(10)])) == 6
    assert order_tags_genre_first([]) == ["Pop"]

def test_energy_duration_scaling():
    assert energy_to_duration_s(0.0) == 120
    assert energy_to_duration_s(1.0) == 240
    assert energy_to_duration_s(0.5) == 180
    assert energy_to_duration_s(-3) == 120  # clamped

def test_steering_prose_synthesizes_seed_fields():
    prose = build_steering_prose({"working_title": "Neon Ache", "mood": "ache",
                                  "energy": 0.9, "placement_hint": "opener"})
    assert "Neon Ache" in prose and "ache" in prose and "high" in prose and "opener" in prose

def test_budget_breaches():
    b = BudgetState(deadline_s=5)
    assert b.consume(0, 0, elapsed_s=3) is None
    assert b.consume(0, 0, elapsed_s=6) == "budget_deadline_exceeded"
