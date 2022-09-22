from itertools import pairwise

import numpy as np
import pandas as pd
import z3
from pipe import map

from dra.datamodels import BlockStats, Stat
from dra.errors import UnsatisfiableModel


class DatabaseConstructionAttack:
    """Database Reconstruction Attacks
    This notebook is adapted from a paper on database reconstruction attacks. You can find the paper here:
    https://cacm.acm.org/magazines/2019/3/234925-understanding-database-reconstruction-attacks-on-public-data/fulltext)

    There are a number of reasons businesses and governments want to share information about people.
    It is important that when sharing information, you consider privacy and anonymity of the people that data is
    derived from. In most cases, aggregate data does little to hinder hackers from being able to re-create a database
    that is either very close, or exactly the same as the original data. In this application,
    we will take a simple example and re-create the database from nothing but aggregate statistics about those people.

    We will mainly use [Z3](https://github.com/Z3Prover/z3) for this.

    Imagine we have the following database that contains information for people within a certain geographic area
    (going forward we refer to this area as a **block**.

    We have *7* people in total in this block. Alongside **age**, we also have each resident's
    **smoking status**, **employment status** and whether they are **married** or not, we publish a variety of
    statistics. You have probably seen something similar in your countries census.

    > 📓 To simplify the example, this fictional world has:
     - Two marriage statuses; Married (**True**) or Single (**False**)
     - Two smoking statuses; Non-Smoker (**False**) or Smoker (**True**)
     - Two employment statuses;  Unemployed (**False**) or Employed (**True**)

    > 👾 One additional piece of logic we know is that any statistics with a count of less than 3 is suppressed
    """

    def __init__(self, stats_file: str, solutions_file: str, min_age: int = 0, max_age: int = 115):
        self.stats: BlockStats = self.read_block_stats(stats_file)
        self.solution: pd.DataFrame = pd.read_csv(solutions_file)
        self.solver = z3.Solver()
        self.min_age = min_age
        self.max_age = max_age
        self.population = range(self.stats.A1.count if self.stats.A1.count is not None else 0)
        self.status: z3.CheckSatResult | None = None
        self.output: pd.DataFrame | None = None

        # Variables
        self.ages: list[z3.IntSort] = z3.IntVector('ages', self.stats.A1.count)

        # TRUE == Married, FALSE = Single
        self.marriages: list[z3.BoolSort] = z3.BoolVector('marriages', self.stats.A1.count)

        # TRUE == Smoker, FALSE = Non-Smoker
        self.smokers: list[z3.BoolSort] = z3.BoolVector('smoker', self.stats.A1.count)

        # TRUE == Employed, FALSE = Unemployed
        self.employed: list[z3.BoolSort] = z3.BoolVector('employed', self.stats.A1.count)

        self.model: z3.ModelRef | None = None

    @staticmethod
    def read_block_stats(file: str) -> BlockStats:
        data = pd.read_csv(file)
        return BlockStats(**data.replace({np.nan: None}).set_index('statistic').to_dict(orient='index'))

    def pairwise_sort_constraint(self, variables: list[z3.Sort]) -> None:
        list(pairwise(variables) | map(lambda pair: self.solver.add(pair[0] <= pair[1])))

    def age_constraints(self) -> None:
        # Constrain each age to our min and max ages
        self.solver.add(*[z3.And(age > self.min_age, age < self.max_age) for age in self.ages])

        # For median, we need the ages sorted by smallest -> largest. Pairwise sort will ensure each element is ordered
        self.pairwise_sort_constraint(self.ages)
        # We can then pluck the middle value and constrain it to our median
        self.solver.add(self.ages[self.stats.A1.count // 2] == self.stats.A1.median)  # type: ignore

        # Averages are quite simple, sum values and divide by our count
        self.solver.add(z3.Sum(self.ages) / self.stats.A1.count == self.stats.A1.mean)

    def marriage_constraints(self) -> None:
        self.solver.add(z3.Sum(self.marriages) == self.stats.B3.count)

        # constrain the ages of married people to the legal age
        self.solver.add(
            *[z3.If(self.marriages[i] == True, self.ages[i] >= 18, self.ages[i] >= 0) for i in self.population]
        )
        # calculate the average for a subset of our database
        self.class_mean(self.ages, True, self.marriages, self.stats.B3)
        self.class_median_cell_size(ages=self.ages, stat=self.stats.B3, class_values=self.marriages, check=True)

        # This is the suppressed statistic, we know that the count must be 0, 1 or 2
        single_count = [z3.If(z3.And(self.marriages[i] == False, self.ages[i] >= 18), 1, 0) for i in self.population]
        self.solver.add(z3.Sum(single_count) >= 0)
        self.solver.add(z3.Sum(single_count) <= 3)

    def smoker_constraints(self) -> None:
        self.solver.add(z3.Sum(self.smokers) == self.stats.B2.count)
        self.class_mean(self.ages, True, self.smokers, self.stats.B2)
        self.class_mean(self.ages, False, self.smokers, self.stats.A2)

        self.class_median_cell_size(ages=self.ages, stat=self.stats.A2, class_values=self.smokers, check=False)
        self.class_median_cell_size(ages=self.ages, stat=self.stats.B2, class_values=self.smokers, check=True)

    def employment_constraints(self) -> None:
        self.solver.add(z3.Sum(self.employed) == self.stats.D2.count)

        self.class_mean(ages=self.ages, check=True, variables=self.employed, stat=self.stats.D2)
        self.class_mean(ages=self.ages, check=False, variables=self.employed, stat=self.stats.C2)

        self.class_median_cell_size(ages=self.ages, stat=self.stats.C2, class_values=self.employed, check=False)
        self.class_median_cell_size(ages=self.ages, stat=self.stats.D2, class_values=self.employed, check=True)

        employee_smoker_count = [
            z3.If(z3.And(self.smokers[i] == False, self.employed[i] == False), 1, 0) for i in self.population
        ]
        self.solver.add(z3.Sum(employee_smoker_count) == self.stats.A4.count)
        self.multi_class_median_cell_size(
            ages=self.ages,
            stat=self.stats.A4,
            class_one_values=self.employed,
            check_one=False,
            class_two_values=self.smokers,
            check_two=False,
        )

    def class_mean(self, ages: list[z3.Sort], check: bool, variables: list[z3.Sort], stat: Stat) -> None:
        self.solver.add(
            z3.Sum([z3.If(variables[i] == check, ages[i], 0) for i in self.population]) / stat.count == stat.mean
        )

    def class_median_cell_size(self, ages: list[z3.Sort], stat: Stat, class_values: list[z3.Sort], check: bool) -> None:
        valid_cell_sizes = (3, 4)
        if stat.count not in valid_cell_sizes:
            raise ValueError(f'cell size must be one of {valid_cell_sizes}, not {stat.count}')
        temp_ids = z3.IntVector(f'{stat.name}-temp-ids', stat.count)
        temp_class_ages = z3.IntVector(f'{stat.name}-temp-class-ages', stat.count)

        self.solver.add(
            z3.If(
                class_values[0] == check,
                temp_ids[0] == 0,
                z3.If(
                    class_values[1] == check,
                    temp_ids[0] == 1,
                    z3.If(
                        class_values[2] == check,
                        temp_ids[0] == 2,
                        z3.If(class_values[3] == check, temp_ids[0] == 3, True),
                    ),
                ),
            )
        )

        self.solver.add(
            z3.If(
                z3.And(class_values[1] == check, temp_ids[0] < 1),
                temp_ids[1] == 1,
                z3.If(
                    z3.And(class_values[2] == check, temp_ids[0] < 2),
                    temp_ids[1] == 2,
                    z3.If(
                        z3.And(class_values[3] == check, temp_ids[0] < 3),
                        temp_ids[1] == 3,
                        z3.If(
                            z3.And(class_values[4] == check, temp_ids[0] < 4),
                            temp_ids[1] == 4,
                            True,
                        ),
                    ),
                ),
            )
        )

        self.solver.add(
            z3.If(
                z3.And(class_values[2] == check, temp_ids[1] < 2),
                temp_ids[2] == 2,
                z3.If(
                    z3.And(class_values[3] == check, temp_ids[1] < 3),
                    temp_ids[2] == 3,
                    z3.If(
                        z3.And(class_values[4] == check, temp_ids[1] < 4),
                        temp_ids[2] == 4,
                        z3.If(
                            z3.And(class_values[5] == check, temp_ids[1] < 5),
                            temp_ids[2] == 5,
                            z3.If(
                                z3.And(class_values[6] == check, temp_ids[1] < 6),
                                temp_ids[2] == 6,
                                True,
                            ),
                        ),
                    ),
                ),
            )
        )

        if stat.count == 4:
            self.solver.add(
                z3.If(
                    z3.And(class_values[3] == check, temp_ids[2] < 3),
                    temp_ids[3] == 3,
                    z3.If(
                        z3.And(class_values[4] == check, temp_ids[2] < 4),
                        temp_ids[3] == 4,
                        z3.If(
                            z3.And(class_values[5] == check, temp_ids[2] < 5),
                            temp_ids[3] == 5,
                            z3.If(
                                z3.And(class_values[6] == check, temp_ids[2] < 6),
                                temp_ids[3] == 6,
                                True,
                            ),
                        ),
                    ),
                )
            )

        self.pairwise_sort_constraint(temp_ids)

        temp_ages = z3.Array(f'{stat.name}-temp-ages', z3.IntSort(), z3.IntSort())
        self.solver.add(*[z3.Select(temp_ages, index) == ages[index] for index in self.population])

        self.solver.add(*[temp_class_ages[i] == z3.Select(temp_ages, temp_ids[i]) for i in range(stat.count)])

        self.pairwise_sort_constraint(temp_class_ages)

        if stat.count == 4:
            self.solver.add((temp_class_ages[1] + temp_class_ages[2]) == stat.median * 2)  # type: ignore
        elif stat.count == 3:
            self.solver.add((temp_class_ages[1]) == stat.median)

    def multi_class_median_cell_size(
        self,
        ages: list[z3.Sort],
        stat: Stat,
        class_one_values: list[z3.Sort],
        class_two_values: list[z3.Sort],
        check_one: bool,
        check_two: bool,
    ):
        valid_cell_sizes = (3, 4)
        if stat.count not in valid_cell_sizes:
            raise ValueError(f'cell size must be one of {valid_cell_sizes}, not {stat.count}')
        temp_ids = z3.IntVector(f'{stat.name}-temp-ids', stat.count)
        temp_class_ages = z3.IntVector(f'{stat.name}-temp-class-ages', stat.count)

        self.solver.add(
            z3.If(
                z3.And(class_one_values[0] == check_one, class_two_values[0] == check_two),
                temp_ids[0] == 0,
                z3.If(
                    z3.And(class_one_values[1] == check_one, class_two_values[1] == check_two),
                    temp_ids[0] == 1,
                    z3.If(
                        z3.And(class_one_values[2] == check_one, class_two_values[2] == check_two),
                        temp_ids[0] == 2,
                        z3.If(
                            z3.And(class_one_values[3] == check_one, class_two_values[3] == check_two),
                            temp_ids[0] == 3,
                            True,
                        ),
                    ),
                ),
            )
        )

        self.solver.add(
            z3.If(
                z3.And(
                    class_one_values[1] == check_one,
                    class_two_values[1] == check_two,
                    temp_ids[0] < 1,
                ),
                temp_ids[1] == 1,
                z3.If(
                    z3.And(
                        class_one_values[2] == check_one,
                        class_two_values[2] == check_two,
                        temp_ids[0] < 2,
                    ),
                    temp_ids[1] == 2,
                    z3.If(
                        z3.And(class_one_values[3] == check_one, class_two_values[3] == check_two),
                        temp_ids[1] == 3,
                        z3.If(
                            z3.And(
                                class_one_values[4] == check_one,
                                class_two_values[4] == check_two,
                                temp_ids[0] < 4,
                            ),
                            temp_ids[1] == 4,
                            True,
                        ),
                    ),
                ),
            )
        )

        self.solver.add(
            z3.If(
                z3.And(
                    class_one_values[2] == check_one,
                    class_two_values[2] == check_two,
                    temp_ids[1] < 2,
                ),
                temp_ids[2] == 2,
                z3.If(
                    z3.And(
                        class_one_values[3] == check_one,
                        class_two_values[3] == check_two,
                        temp_ids[1] < 3,
                    ),
                    temp_ids[2] == 3,
                    z3.If(
                        z3.And(
                            class_one_values[4] == check_one,
                            class_two_values[4] == check_two,
                            temp_ids[1] < 4,
                        ),
                        temp_ids[2] == 4,
                        z3.If(
                            z3.And(
                                class_one_values[5] == check_one,
                                class_two_values[5] == check_two,
                                temp_ids[1] < 5,
                            ),
                            temp_ids[2] == 5,
                            z3.If(
                                z3.And(
                                    class_one_values[6] == check_one,
                                    class_two_values[6] == check_two,
                                    temp_ids[1] < 6,
                                ),
                                temp_ids[2] == 6,
                                True,
                            ),
                        ),
                    ),
                ),
            )
        )

        if stat.count == 4:
            self.solver.add(
                z3.If(
                    z3.And(
                        class_one_values[3] == check_one,
                        class_two_values[3] == check_two,
                        temp_ids[2] < 3,
                    ),
                    temp_ids[3] == 3,
                    z3.If(
                        z3.And(
                            class_one_values[4] == check_one,
                            class_two_values[4] == check_two,
                            temp_ids[2] < 4,
                        ),
                        temp_ids[3] == 4,
                        z3.If(
                            z3.And(
                                class_one_values[5] == check_one,
                                class_two_values[5] == check_two,
                                temp_ids[2] < 5,
                            ),
                            temp_ids[3] == 5,
                            z3.If(
                                z3.And(
                                    class_one_values[6] == check_one,
                                    class_two_values[6] == check_two,
                                    temp_ids[2] < 6,
                                ),
                                temp_ids[3] == 6,
                                True,
                            ),
                        ),
                    ),
                )
            )

        self.pairwise_sort_constraint(temp_ids)

        temp_ages = z3.Array(f'{stat.name}-temp-ages', z3.IntSort(), z3.IntSort())
        self.solver.add(*[z3.Select(temp_ages, index) == ages[index] for index in self.population])

        self.solver.add(*[temp_class_ages[i] == z3.Select(temp_ages, temp_ids[i]) for i in range(stat.count)])

        self.pairwise_sort_constraint(temp_class_ages)

        if stat.count == 4:
            self.solver.add((temp_class_ages[1] + temp_class_ages[2]) == stat.median * 2)  # type: ignore
        elif stat.count == 3:
            self.solver.add((temp_class_ages[1]) == stat.median)

    def check_accuracy(self) -> float:
        match, non_match = 0, 0
        computed = [tuple(v.values()) for v in self.output.to_dict(orient='records')]  # type: ignore
        original = [tuple(v.values()) for v in self.solution.to_dict(orient='records')]  # type: ignore

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
                    'age': [self.model[v] for v in self.ages],
                    'married': [self.model[v] for v in self.marriages],
                    'smoker': [self.model[v] for v in self.smokers],
                    'employed': [self.model[v] for v in self.employed],
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
