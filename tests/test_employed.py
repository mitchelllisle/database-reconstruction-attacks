from math import floor


def test_employment_count(employments, stats):
    assert len(employments.employed) == stats.D2.count
    assert len(employments.unemployed) == stats.C2.count


def test_employment_mean(employments, stats):
    assert floor(employments.employed.age.mean()) == stats.D2.mean
    assert floor(employments.unemployed.age.mean()) == stats.C2.mean


def test_employment_median(employments, stats):
    assert floor(employments.employed.age.median()) == stats.D2.median
    assert floor(employments.unemployed.age.median()) == stats.C2.median
