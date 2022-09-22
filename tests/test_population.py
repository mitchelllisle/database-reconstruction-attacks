from math import floor

import pandas as pd

from dra.attack import DatabaseConstructionAttack


def test_population_count(dra: DatabaseConstructionAttack, stats):
    assert len(dra.ages) == stats.A1.count


def test_population_mean(model: pd.DataFrame, stats):
    assert floor(model.age.mean()) == stats.A1.mean


def test_population_median(model: pd.DataFrame, stats):
    assert floor(model.age.median()) == stats.A1.median
