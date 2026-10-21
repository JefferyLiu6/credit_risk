from credit_risk_champion_challenger.service import risk_band


def test_risk_bands_cover_boundary_values():
    assert risk_band(0.01) == "low"
    assert risk_band(0.05) == "moderate"
    assert risk_band(0.10) == "high"
    assert risk_band(0.20) == "very_high"

