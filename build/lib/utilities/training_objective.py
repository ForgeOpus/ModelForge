from enum import Enum


class TrainingObjective(str, Enum):
    SUPERVISED = "supervised"
    PREFERENCE = "preference"
    REINFORCEMENT = "reinforcement"
