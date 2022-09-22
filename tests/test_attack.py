import pytest

from dra.__main__ import main
from dra.attack import DatabaseConstructionAttack
from dra.errors import UnsatisfiableModel


def test_main(args):
    main(args)


def test_output(dra, model):
    assert dra.check_accuracy() >= 92


def test_unsat(args):
    attack = DatabaseConstructionAttack(stats_file=args.input, solutions_file=args.solution, max_age=50)
    with pytest.raises(UnsatisfiableModel):
        attack.run()


def test_unstat_as_df(dra):
    with pytest.raises(TypeError):
        dra.model_as_dataframe()
