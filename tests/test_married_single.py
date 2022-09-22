from math import floor

import pandas as pd

from dra.database import stats


def test_marriages_count(marriages):
    assert len(marriages.married) == stats.B3.count
    assert len(marriages.single) > 0
    assert len(marriages.single) <= 3


def test_marriages_mean(marriages):
    assert floor(marriages.married.age.mean()) == stats.B3.mean


def test_population_median(marriages: pd.DataFrame):
    assert floor(marriages.married.age.median()) == stats.B3.median
