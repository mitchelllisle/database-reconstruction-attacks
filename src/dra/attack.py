from typing import Optional, Tuple

import numpy as np
import pandas as pd
import z3

from dra.datamodels import BlockStats
from dra.errors import UnsatisfiableModel


class DatabaseReconstructionAttack:
    """Database Reconstruction Attacks
    This notebook is adapted from a paper on database reconstruction attacks. You can find the paper
    [here](
    https://cacm.acm.org/magazines/2019/3/234925-understanding-database-reconstruction-attacks-on-public-data/fulltext
    )

    There are a number of reasons businesses and governments want to share information about people.
    It is important that when sharing information, you consider privacy and anonymity of the people that data is
    derived from. In most cases, aggregate data does little to hinder hackers from being able to re-create a database
    that is either very close, or exactly the same as the original data. In this application,
    we will take a simple example and re-create the database from nothing but aggregate statistics about those people.

    We will mainly use [Z3](https://github.com/Z3Prover/z3) for this.
    Imagine we have the following database that contains information for people within a certain geographic area
    (going forward we refer to this area as a **block**.)

    We have *7* people in total in this block. Alongside **age**, we also have each resident's
    **smoking status**, **employment status** and whether they are **married** or not, we publish a variety of
    statistics. You have probably seen something similar in your countries census.

    > 📓 To simplify the example, this fictional world has:
     - Two marriage statuses; Married (**True**) or Single (**False**)
     - Two smoking statuses; Non-Smoker (**False**) or Smoker (**True**)
     - Two employment statuses;  Unemployed (**False**) or Employed (**True**)

    > 👾 One additional piece of logic we know is that any statistics with a count of less than 3 is suppressed
    """

    def __init__(self, stats_file: str, database_file: str, min_age: int = 0, max_age: int = 115):
        self.stats: BlockStats = self.read_block_stats(stats_file)
        self.database: pd.DataFrame = pd.read_csv(database_file)
        self.solver: z3.Solver = z3.Solver()
        self.min_age: int = min_age
        self.max_age: int = max_age
        self.population: range = range(self.stats.A1.count if self.stats.A1.count is not None else 0)
        self.status: Optional[z3.CheckSatResult] = None
        self.output: Optional[pd.DataFrame] = None

        # Variables
        self.ages: z3.ArraySort = z3.Array('ages', z3.IntSort(), z3.IntSort())

        self.married_indices, self.single_indices = self.split_pair_of_indices(
            name_pair=('married', 'single'), first_count=self.stats.B3.count or 0
        )

        self.smoker_indices, self.non_smoker_indices = self.split_pair_of_indices(
            name_pair=('smoker', 'non_smoker'), first_count=self.stats.B2.count or 0
        )

        self.employed_indices, self.unemployed_indices = self.split_pair_of_indices(
            name_pair=('employed', 'unemployed'), first_count=self.stats.D2.count or 0
        )

        self.unemployed_non_smoker_indices, _ = self.split_pair_of_indices(
            name_pair=('unemployed_non_smoker', 'rest'), first_count=self.stats.A4.count or 0
        )

        self.model: Optional[z3.ModelRef] = None

    @staticmethod
    def read_block_stats(file: str) -> BlockStats:
        data = pd.read_csv(file)
        return BlockStats(**data.replace({np.nan: None}).set_index('statistic').to_dict(orient='index'))

    def add_sorted_constraint_for_ages(self):
        for pair in zip(self.population[:-1], self.population[1:]):
            self.solver.add(z3.Select(self.ages, pair[0]) <= z3.Select(self.ages, pair[1]))

    def split_pair_of_indices(self, name_pair: Tuple[str, str], first_count: int):
        first_indices = z3.IntVector(name_pair[0], first_count)
        last_indices = z3.IntVector(name_pair[1], len(self.population) - first_count)

        # indices must between 0 and 7
        self.solver.add(*[z3.And(idx >= 0, idx < len(self.population)) for idx in first_indices + last_indices])

        # indices must be distinct
        self.solver.add(z3.Distinct(*[idx for idx in first_indices + last_indices]))

        # indices must be sorted
        for pair in zip(range(first_count)[:-1], range(first_count)[1:]):
            self.solver.add(first_indices[pair[0]] < first_indices[pair[1]])

        for pair in zip(range(7 - first_count)[:-1], range(7 - first_count)[1:]):
            self.solver.add(last_indices[pair[0]] < last_indices[pair[1]])

        return first_indices, last_indices

    def add_median_constraint(self, indices, median):
        med_idx = len(indices) // 2

        if len(indices) % 2 == 0:
            self.solver.add(self.ages[indices[med_idx - 1]] + self.ages[indices[med_idx]] == median * 2)
        else:
            self.solver.add(z3.Store(self.ages, indices[med_idx], median) == self.ages)

    def add_mean_constraint(self, indices, mean):
        self.solver.add(z3.Sum([self.ages[idx] for idx in indices]) / float(len(indices)) == mean)

    def age_constraints(self) -> None:
        # Constrain each age to our min and max ages
        self.solver.add(
            *[
                z3.And(z3.Select(self.ages, i) > self.min_age, z3.Select(self.ages, i) < self.max_age)
                for i in self.population
            ]
        )

        # For median we need the ages sorted by smallest -> largest
        self.add_sorted_constraint_for_ages()

        # We can then pluck the middle value and constrain it to our median
        self.solver.add(z3.Store(self.ages, len(self.population) // 2, self.stats.A1.median) == self.ages)

        # Averages are quite simple, sum values and divide by our count
        age_sum = z3.Sum([z3.Select(self.ages, i) for i in self.population]) / float(len(self.population))
        self.solver.add(age_sum == self.stats.A1.mean)

    def marriage_constraints(self) -> None:

        # constrain the ages of married people to the legal age
        self.solver.add(*[self.ages[idx] >= 18 for idx in self.married_indices])
        self.solver.add(*[self.ages[idx] >= 0 for idx in self.single_indices])

        # calculate the average for a subset of our database
        self.add_mean_constraint(indices=self.married_indices, mean=self.stats.B3.mean)

        # calculate the median for a subset of our database
        self.add_median_constraint(indices=self.married_indices, median=self.stats.B3.median)

        # This is the supressed statistic, we know that the count must be 0, 1 or 2
        single_adult_count = [z3.If(self.ages[idx] >= 18, 1, 0) for idx in self.single_indices]
        self.solver.add(z3.Sum(single_adult_count) >= 0)
        self.solver.add(z3.Sum(single_adult_count) <= 2)

    def smoker_constraints(self) -> None:

        # add mean constraints
        self.add_mean_constraint(indices=self.smoker_indices, mean=self.stats.B2.mean)
        self.add_mean_constraint(indices=self.non_smoker_indices, mean=self.stats.A2.mean)

        # add median constraints
        self.add_median_constraint(indices=self.smoker_indices, median=self.stats.B2.median)
        self.add_median_constraint(indices=self.non_smoker_indices, median=self.stats.A2.median)

    def employment_constraints(self) -> None:

        # add mean constraints
        self.add_mean_constraint(indices=self.employed_indices, mean=self.stats.D2.mean)
        self.add_mean_constraint(indices=self.unemployed_indices, mean=self.stats.C2.mean)

        # add median constraints
        self.add_median_constraint(indices=self.employed_indices, median=self.stats.D2.median)
        self.add_median_constraint(indices=self.unemployed_indices, median=self.stats.C2.median)

        # intersection of umemployed and non-smoker
        self.solver.add(
            *[
                z3.And(
                    z3.Or(*[i == idx for i in self.unemployed_indices]),
                    z3.Or(*[j == idx for j in self.non_smoker_indices]),
                )
                for idx in self.unemployed_non_smoker_indices
            ]
        )

        # add mean constraints
        self.add_mean_constraint(indices=self.unemployed_non_smoker_indices, mean=self.stats.A4.mean)

        # add median constraints
        self.add_median_constraint(indices=self.unemployed_non_smoker_indices, median=self.stats.A4.median)

    def check_accuracy(self) -> float:
        match, non_match = 0, 0
        computed = [tuple(v.values()) for v in self.output.to_dict(orient='records')]  # type: ignore
        original = [tuple(v.values()) for v in self.database.to_dict(orient='records')]  # type: ignore

        to_check = [list(zip(computed[i], original[i])) for i in self.population]
        for items in to_check:
            for pair in items:
                if pair[0] == pair[1]:
                    match += 1
                else:
                    non_match += 1
        return (match / (match + non_match)) * 100

    def model_as_dataframe(self) -> pd.DataFrame:
        if isinstance(self.model, z3.ModelRef):
            df = pd.DataFrame(
                {
                    'age': [self.model.evaluate(z3.Select(self.ages, i)) for i in self.population],
                    'married': [i in [self.model[idx] for idx in self.married_indices] for i in self.population],
                    'smoker': [i in [self.model[idx] for idx in self.smoker_indices] for i in self.population],
                    'employed': [i in [self.model[idx] for idx in self.employed_indices] for i in self.population],
                }
            )
            return df
        raise TypeError(
            'model is not initialised. Either there is no valid model, '
            'or you forgot to run DatabaseReconstructionAttack.run?'
        )

    def run(self) -> pd.DataFrame:
        """Run & Reconstruct the Database
        With this model we're able to reconstruct between **92-100%** of the database depending on the solution the
        solver picks. The solver comes up with 3-4 solutions that fit our constraints and in this sample we can compare
        it with the original database to see how far off we are.

        Pretty remarkable results given we **only used summary statistics** (and included a suppressed statistic!)

        Being aware of these types of attacks is important to ensure you design and build shared data that protects
        against a motivated attacker that can use these techniques to reconstruct your original source data.

        > ✨ Hint: See [Differential Privacy](https://desfontain.es/privacy/differential-privacy-awesomeness.html) for
        more on how to do this
        """
        self.age_constraints()
        self.marriage_constraints()
        self.smoker_constraints()
        self.employment_constraints()

        self.status = self.solver.check()
        if self.status == z3.sat:
            self.model = self.solver.model()
            self.output = self.model_as_dataframe()
            return self.output
        raise UnsatisfiableModel('No valid model found')
