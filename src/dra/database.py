import pandas as pd

from dra.datamodels import BlockStats, Stat

solution = pd.DataFrame(
    [
        (8, False, False, False),
        (18, False, True, True),
        (24, False, False, True),
        (30, True, True, True),
        (36, True, False, False),
        (66, True, False, False),
        (84, True, True, False),
    ],
    columns=['age', 'married', 'smoker', 'employed'],
)

stats = BlockStats(
    A1=Stat(name='total-population', count=7, median=30, mean=38),
    A2=Stat(name='non-smoker', count=4, median=30, mean=33.5),
    B2=Stat(name='smoker', count=3, median=30, mean=44),
    C2=Stat(name='unemployed', count=4, median=51, mean=48.5),
    D2=Stat(name='employed', count=3, median=24, mean=24),
    A3=Stat(name='single-adults', count=None, median=None, mean=None),
    B3=Stat(name='married-adults', count=4, median=51, mean=54),
    A4=Stat(name='unemployed-non-smoker', count=3, median=36, mean=37),
)
