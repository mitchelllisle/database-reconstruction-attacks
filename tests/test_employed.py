from math import floor

from dra.database import stats


def test_employment_count(employments):
    assert len(employments.employed) == stats.D2.count
    assert len(employments.unemployed) == stats.C2.count


def test_employment_mean(employments):
    assert floor(employments.employed.age.mean()) == stats.D2.mean
    assert floor(employments.unemployed.age.mean()) == stats.C2.mean


def test_employment_median(employments):
    assert floor(employments.employed.age.median()) == stats.D2.median
    assert floor(employments.unemployed.age.median()) == stats.C2.median
