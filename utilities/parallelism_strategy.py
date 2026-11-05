from enum import Enum


class ParallelismStrategy(str, Enum):
    DATA_PARALLEL = "data_parallel"
    FSDP = "fsdp"
