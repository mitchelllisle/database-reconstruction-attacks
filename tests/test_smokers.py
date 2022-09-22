from math import floor

from dra.database import stats


def test_marriages_count(smokers):
    assert len(smokers.smoker) == stats.B2.count
    assert len(smokers.non_smoker) == stats.A2.count


def test_marriages_mean(smokers):
    assert floor(smokers.smoker.age.mean()) == stats.B2.mean
    assert floor(smokers.non_smoker.age.mean()) == stats.A2.mean


def test_population_median(smokers):
    assert floor(smokers.smoker.age.median()) == stats.B2.median
    assert floor(smokers.non_smoker.age.median()) == stats.A2.median
