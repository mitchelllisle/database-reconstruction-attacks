from math import floor


def test_unemployed_non_smoker_count(unemployed_non_smoker, stats):
    assert len(unemployed_non_smoker) == stats.A4.count


def test_unemployed_non_smoker_mean(unemployed_non_smoker, stats):
    assert floor(unemployed_non_smoker.age.mean()) >= stats.A4.mean - 1
    assert floor(unemployed_non_smoker.age.mean()) <= stats.A4.mean


def test_unemployed_non_smoker_median(unemployed_non_smoker, stats):
    assert floor(unemployed_non_smoker.age.median()) == stats.A4.median
