import numpy as np
import pandas as pd

from credit_risk_champion_challenger.monitoring import categorical_psi, drift_band, numeric_psi


def test_identical_numeric_population_is_stable():
    values = np.arange(100, dtype=float)
    value = numeric_psi(values, values)
    assert value == 0
    assert drift_band(value) == "stable"


def test_shifted_numeric_population_is_flagged():
    reference = np.linspace(0, 1, 500)
    current = np.linspace(2, 3, 500)
    assert drift_band(numeric_psi(reference, current)) == "material"


def test_categorical_psi_handles_new_categories_and_missing_values():
    reference = pd.Series(["a", "a", "b", None])
    current = pd.Series(["a", "c", "c", None])
    assert categorical_psi(reference, current) > 0

