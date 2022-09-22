from pydantic import BaseModel


class Stat(BaseModel):
    name: str
    count: int | None
    median: int | None
    mean: int | float | None


class BlockStats(BaseModel):
    A1: Stat
    A2: Stat
    B2: Stat
    C2: Stat
    D2: Stat
    A3: Stat
    B3: Stat
    A4: Stat
