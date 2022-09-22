import pytest

from dra.__main__ import main
from dra.attack import DatabaseConstructionAttack
from dra.database import solution, stats
from dra.errors import UnsatisfiableModel


def test_main():
    main()


def test_output(dra, model):
    assert dra.check_accuracy(model, solution) >= 92


def test_unsat():
    attack = DatabaseConstructionAttack(stats, max_age=50)
    with pytest.raises(UnsatisfiableModel):
        attack.run()


def test_unstat_as_df(dra):
    with pytest.raises(TypeError):
        dra.model_as_dataframe()
