from dataclasses import dataclass

import pandas as pd
import pytest

from dra.attack import DatabaseConstructionAttack
from dra.datamodels import BlockStats


class Args:
    @property
    def input(self) -> str:
        return 'data/block-stats.csv'

    @property
    def solution(self) -> str:
        return 'data/solution.csv'

    @property
    def output(self) -> str:
        return 'data/reconstruction.csv'

    @property
    def min_age(self) -> int:
        return 0

    @property
    def max_age(self) -> int:
        return 115


@dataclass
class MarriageClasses:
    married: pd.DataFrame
    single: pd.DataFrame


@dataclass
class SmokersClasses:
    smoker: pd.DataFrame
    non_smoker: pd.DataFrame


@dataclass
class EmploymentClasses:
    employed: pd.DataFrame
    unemployed: pd.DataFrame


@pytest.fixture
def args():
    return Args()


@pytest.fixture
def dra(args) -> DatabaseConstructionAttack:
    attack = DatabaseConstructionAttack(stats_file=args.input, solutions_file=args.solution)
    return attack


@pytest.fixture
def stats(args) -> BlockStats:
    attack = DatabaseConstructionAttack(stats_file=args.input, solutions_file=args.solution)
    return attack.stats


@pytest.fixture
def model(dra) -> pd.DataFrame:
    dra.run()
    model = dra.model_as_dataframe()
    model.age = model.age.astype(str).astype(int)
    model.married = model.married.apply(lambda row: True if row == True else False)
    model.smoker = model.smoker.apply(lambda row: True if row == True else False)
    model.employed = model.employed.apply(lambda row: True if row == True else False)
    return model


@pytest.fixture
def marriages(model: pd.DataFrame) -> MarriageClasses:
    return MarriageClasses(married=model[model.married == True], single=model[model.married == False])


@pytest.fixture
def smokers(model: pd.DataFrame) -> SmokersClasses:
    return SmokersClasses(smoker=model[model.smoker == True], non_smoker=model[model.smoker == False])


@pytest.fixture
def employments(model: pd.DataFrame) -> EmploymentClasses:
    return EmploymentClasses(employed=model[model.employed == True], unemployed=model[model.employed == False])


@pytest.fixture
def unemployed_non_smoker(model: pd.DataFrame) -> pd.DataFrame:
    return model[(model.employed == False) & (model.smoker == False)]
