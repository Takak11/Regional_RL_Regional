from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:

    # EXPECTED_MAX_QUEUING_LENGTH = 3.0
    EXPECTED_MAX_FCS_WAIT_TIME = 15.0

    # 基本设置
    FCS_CHARGING_PILES: int = 3
    MCS_PER_REGION: int = 8

    TIME_STEP: int = 5
    TOTAL_TIMESTEP: int = 100
    MAX_WAITING_TIME: int = 15

    MCS_SCHEDULE_R: int = 3

    MOVING_SPEED: float = 40.0
    # EV电量相关
    CHARGING_POWER: float = 50.0
    MCS_CHARGING_POWER: float = 100.0
    TARGET_CHARGE_LEVEL: float = 0.85
    REQUEST_THRESHOLD: float = 0.15
    SAFE_REACH_THRESHOLD: float = 1
    BATTERY_CAPACITY: float = 60.0
    ENERGY_CONSUMPTION: float = 0.16
    CHARGING_FEE: float = 0.8

    distribution_mode: str = 'normal'
    mean_percentage: float = 0.8
    std_percentage: float = 0.28
    min_percentage: float = 0.1
    max_percentage: float = 0.8

    random_seed: Optional[int] = 42



