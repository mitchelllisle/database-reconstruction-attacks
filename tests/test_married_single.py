from math import floor


def test_marriages_count(marriages, stats):
    assert len(marriages.married) == stats.B3.count
    assert len(marriages.single) > 0
    assert len(marriages.single) <= 3


def test_marriages_mean(marriages, stats):
    assert floor(marriages.married.age.mean()) == stats.B3.mean


def test_marriages_median(marriages, stats):
    assert floor(marriages.married.age.median()) == stats.B3.median
