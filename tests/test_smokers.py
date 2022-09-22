from math import floor


def test_smokers_count(smokers, stats):
    assert len(smokers.smoker) == stats.B2.count
    assert len(smokers.non_smoker) == stats.A2.count


def test_smokers_mean(smokers, stats):
    assert floor(smokers.smoker.age.mean()) == stats.B2.mean
    assert floor(smokers.non_smoker.age.mean()) == stats.A2.mean


def test_smokers_median(smokers, stats):
    assert floor(smokers.smoker.age.median()) == stats.B2.median
    assert floor(smokers.non_smoker.age.median()) == stats.A2.median
