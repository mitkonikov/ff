from enum import Enum


class SparsityType(Enum):
    HOYER = 1
    L1_NEG_ENTROPY = 2
    L2_NEG_ENTROPY = 3
    GINI = 4
