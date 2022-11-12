import pytest

from dra.__main__ import run
from dra.attack import DatabaseReconstructionAttack
from dra.errors import UnsatisfiableModel


def test_main(args):
    run(args)


def test_output(dra, model):
    assert dra.check_accuracy() >= 92


def test_unsat(args):
    attack = DatabaseReconstructionAttack(stats_file=args.input, database_file=args.database, max_age=50)
    with pytest.raises(UnsatisfiableModel):
        attack.run()


def test_unstat_as_df(dra):
    with pytest.raises(TypeError):
        dra.model_as_dataframe()
