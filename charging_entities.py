import numpy as np

from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from params_config import Config

from dataclasses import dataclass
from typing import Tuple, Optional
from datetime import datetime
from enum import Enum


config = Config()


@dataclass
class EV:
    id: str
    current_location: Tuple[float, float]  # (lat, lon)
    target_location: Tuple[float, float]
    current_charge: float  # kWh
    region_id: int
    status: str = 'running'
    charging_request_time: Optional[datetime] = None
    trajectory: List[Tuple[float, float, datetime]] = None

    # 新增属性用于边缘环境
    trajectory_index: int = 0  # 当前轨迹索引
    charging_history: List[Dict] = field(default_factory=list)  # 充电历史
    total_travel_distance: float = 0.0  # 总行驶距离
    assigned_mcs: Optional[int] = None  # 分配的MCS
    assigned_fcs: Optional[int] = None  # 分配的FCS

    def needs_charging(self) -> bool:
        return self.current_charge < config.BATTERY_CAPACITY * 0.2

    def charge_needed(self) -> float:
        """需要充电的电量（到80%）"""
        target_charge = config.BATTERY_CAPACITY * config.TARGET_CHARGE_LEVEL
        return max(0, target_charge - self.current_charge)


class MCSStatus(Enum):
    IDLE = "idle"  # 空闲
    CHARGING = "charging"  # 充电中
    MOVING = "moving"  # 移动中
    ASSIGNED = "assigned"  # 已分配但未到达

@dataclass
class MCS:
    """移动充电站(Mobile Charging Station)实体类"""

    # 基本属性
    id: str
    current_location: Tuple[float, float]  # (纬度, 经度)
    region_id: int

    # 状态属性
    is_busy: bool = False
    assigned_ev: Optional[str] = None
    income: float = 0.0
    status: MCSStatus = MCSStatus.IDLE

    # 移动相关属性
    target_location: Optional[Tuple[float, float]] = None
    arrival_time: Optional[datetime] = None
    movement_speed: float = 30.0  # km/h，移动速度
    cost_per_km: float = 0.5  # 每公里移动成本
    total_distance_traveled: float = 0.0  # 总移动距离

    # 充电相关属性
    charging_start_time: Optional[datetime] = None
    estimated_finish_time: Optional[datetime] = None
    charge_amount: float = 0.0  # 当前充电量
    efficiency: float = 0.9  # 充电效率

    def __post_init__(self):
        """初始化后处理"""
        # 确保assigned_ev为字符串类型
        if self.assigned_ev is not None and not isinstance(self.assigned_ev, str):
            self.assigned_ev = str(self.assigned_ev)

    def is_available(self) -> bool:
        """检查MCS是否可用"""
        return self.status == MCSStatus.IDLE and not self.is_busy

    def get_utilization_info(self) -> dict:
        """获取MCS利用率信息"""
        return {
            'id': self.id,
            'status': self.status.value,
            'region_id': self.region_id,
            'total_distance': self.total_distance_traveled,
            'total_income': self.income,
            'efficiency': self.efficiency,
            'current_location': self.current_location
        }

    def reset_charging_state(self):
        """重置充电状态"""
        self.charging_start_time = None
        self.estimated_finish_time = None
        self.charge_amount = 0.0
        self.assigned_ev = None
        self.status = MCSStatus.IDLE
        self.is_busy = False

    def start_movement(self, target: Tuple[float, float],
                       arrival_time: datetime, assigned_ev_id: Optional[str] = None):
        """开始移动到目标位置"""
        self.status = MCSStatus.MOVING
        self.target_location = target
        self.arrival_time = arrival_time
        self.assigned_ev = assigned_ev_id if assigned_ev_id else None
        self.is_busy = True

    def complete_movement(self):
        """完成移动"""
        if self.target_location:
            self.current_location = self.target_location
            self.target_location = None
            self.arrival_time = None
            self.status = MCSStatus.ASSIGNED

    def start_charging(self, charge_amount: float, start_time: datetime,
                       finish_time: datetime):
        """开始充电"""
        self.status = MCSStatus.CHARGING
        self.charge_amount = charge_amount
        self.charging_start_time = start_time
        self.estimated_finish_time = finish_time
        self.is_busy = True

    def complete_charging(self) -> float:
        """完成充电，返回收入"""
        income = self.charge_amount * 0.8  # 假设每kWh收费0.8元
        self.income += income
        self.reset_charging_state()
        return income


@dataclass
class ChargingPile:
    """FCS充电桩"""
    pile_id: int
    is_occupied: bool = False
    ev_id: Optional[int] = None
    charging_start_time: Optional[datetime] = None
    estimated_finish_time: Optional[datetime] = None
    charge_amount: float = 0.0


@dataclass
class QueuedRequest:
    """排队请求信息"""
    ev_id: int
    request_time: datetime
    estimated_charge_needed: float
    estimated_charge_time: float  # 预计充电时长(分钟)
    expected_start_time: datetime  # 预计开始充电时间
    assigned_pile_id: Optional[int] = None  # 预分配的充电桩ID


@dataclass
class FCS:
    id: int
    location: Tuple[float, float]
    region_id: int
    charging_piles: List[ChargingPile] = field(default_factory=list)
    waiting_queue: List[QueuedRequest] = field(default_factory=list)  # 等待队列

    def __post_init__(self):
        if not self.charging_piles:
            self.charging_piles = [
                ChargingPile(pile_id=i) for i in range(config.FCS_CHARGING_PILES)
            ]

    def get_available_piles(self) -> List[ChargingPile]:
        """获取可用充电桩"""
        return [pile for pile in self.charging_piles if not pile.is_occupied]

    def get_queue_length(self) -> int:
        """获取排队长度"""
        return len(self.waiting_queue)

    def _get_pile_available_times(self, current_time: datetime) -> Dict[int, datetime]:
        """获取每个充电桩的可用时间"""
        pile_times = {}

        for pile in self.charging_piles:
            if not pile.is_occupied:
                # 空闲桩，立即可用
                pile_times[pile.pile_id] = current_time
            elif pile.estimated_finish_time:
                # 占用中，使用预计完成时间
                pile_times[pile.pile_id] = pile.estimated_finish_time
            else:
                # 异常情况，假设立即可用
                pile_times[pile.pile_id] = current_time

        return pile_times

    def _recalculate_queue_schedule(self, current_time: datetime):
        """重新计算整个队列的调度"""
        if not self.waiting_queue:
            return

        # 获取每个桩的当前可用时间
        pile_available_times = self._get_pile_available_times(current_time)

        # 为队列中的每个请求重新分配桩和开始时间
        for request in self.waiting_queue:
            # 找到最早可用的桩
            earliest_pile_id = min(pile_available_times,
                                   key=lambda pid: pile_available_times[pid])
            earliest_time = pile_available_times[earliest_pile_id]

            # 分配给这个请求
            request.assigned_pile_id = earliest_pile_id
            request.expected_start_time = earliest_time

            # 更新该桩的下次可用时间
            pile_available_times[earliest_pile_id] = earliest_time + timedelta(
                minutes=request.estimated_charge_time
            )

    def add_to_queue(self, ev_id: int, request_time: datetime,
                     charge_needed: float, current_time: datetime) -> QueuedRequest:
        """将EV加入排队队列"""
        # 计算充电时长
        charge_time = (charge_needed / config.CHARGING_POWER) * 60  # 分钟

        # 创建排队请求（先不分配桩和时间）
        queued_request = QueuedRequest(
            ev_id=ev_id,
            request_time=request_time,
            estimated_charge_needed=charge_needed,
            estimated_charge_time=charge_time,
            expected_start_time=current_time  # 临时值
        )

        self.waiting_queue.append(queued_request)

        # 重新计算整个队列的调度
        self._recalculate_queue_schedule(current_time)

        return queued_request

    def remove_from_queue(self, ev_id: int, current_time: datetime) -> bool:
        """从队列中移除EV并重新调度"""
        # 查找并移除
        removed = False
        for i, req in enumerate(self.waiting_queue):
            if req.ev_id == ev_id:
                self.waiting_queue.pop(i)
                removed = True
                break

        if not removed:
            return False

        # 重新计算整个队列的调度
        self._recalculate_queue_schedule(current_time)

        return True

    def on_pile_released(self, pile_id: int, current_time: datetime):
        """当充电桩释放时调用，重新调度队列"""
        # 充电桩释放，可能影响队列调度
        self._recalculate_queue_schedule(current_time)

    def get_next_from_queue(self, current_time: datetime) -> Optional[Tuple[int, int]]:
        """从队列中获取下一个要充电的EV及其分配的桩

        Returns:
            (ev_id, pile_id) 或 None
        """
        if not self.waiting_queue:
            return None

        # 重新计算调度（确保使用最新状态）
        self._recalculate_queue_schedule(current_time)

        # 找出所有可以立即开始充电的请求
        ready_requests = []
        for i, req in enumerate(self.waiting_queue):
            if req.expected_start_time <= current_time and req.assigned_pile_id is not None:
                # 检查分配的桩是否真的可用
                pile = self.charging_piles[req.assigned_pile_id]
                if not pile.is_occupied:
                    ready_requests.append((i, req))

        if not ready_requests:
            return None

        # 选择等待时间最长的（FIFO原则）
        idx, selected_req = min(ready_requests,
                                key=lambda x: x[1].request_time)

        # 从队列中移除
        self.waiting_queue.pop(idx)

        # 重新调度剩余队列
        if self.waiting_queue:
            self._recalculate_queue_schedule(current_time)

        return (selected_req.ev_id, selected_req.assigned_pile_id)

    def get_estimated_wait_time(self, current_time: datetime = None) -> float:
        """获取新到达EV的预计等待时间(分钟)"""
        if current_time is None:
            raise ValueError("必须提供current_time参数")

        # 如果有可用充电桩,无需等待
        if self.get_available_piles():
            return 0.0

        # 获取每个桩的可用时间
        pile_available_times = self._get_pile_available_times(current_time)

        # 考虑队列中的请求，模拟调度
        temp_times = pile_available_times.copy()

        for request in self.waiting_queue:
            # 找最早可用的桩
            earliest_time = min(temp_times.values())
            earliest_pile = min(temp_times, key=lambda pid: temp_times[pid])

            # 更新该桩的可用时间
            temp_times[earliest_pile] = earliest_time + timedelta(
                minutes=request.estimated_charge_time
            )

        # 新请求将在最早可用的桩上充电
        earliest_available = min(temp_times.values())
        wait_time = (earliest_available - current_time).total_seconds() / 60

        return max(0, wait_time)

    def get_queue_info(self) -> List[Dict]:
        """获取队列信息"""
        return [
            {
                'ev_id': req.ev_id,
                'expected_start_time': req.expected_start_time,
                'estimated_charge_time': req.estimated_charge_time,
                'request_time': req.request_time,
                'assigned_pile_id': req.assigned_pile_id,
                'wait_time_so_far': (datetime.now() - req.request_time).total_seconds() / 60
            }
            for req in self.waiting_queue
        ]

    def get_pile_schedule(self, current_time: datetime) -> Dict[int, List[Dict]]:
        """获取每个充电桩的调度情况"""
        schedule = {pile.pile_id: [] for pile in self.charging_piles}

        # 添加当前正在充电的
        for pile in self.charging_piles:
            if pile.is_occupied and pile.ev_id:
                schedule[pile.pile_id].append({
                    'ev_id': pile.ev_id,
                    'start_time': pile.charging_start_time,
                    'end_time': pile.estimated_finish_time,
                    'status': 'charging'
                })

        # 添加排队等待的
        for req in self.waiting_queue:
            if req.assigned_pile_id is not None:
                schedule[req.assigned_pile_id].append({
                    'ev_id': req.ev_id,
                    'start_time': req.expected_start_time,
                    'end_time': req.expected_start_time + timedelta(
                        minutes=req.estimated_charge_time
                    ),
                    'status': 'queued'
                })

        return schedule

@dataclass
class ChargingRequest:
    ev_id: int
    request_time: datetime
    location: Tuple[float, float]
    current_charge: float
    region_id: int
    can_reach_fcs: bool  # 是否能安全到达FCS
    estimated_charge_needed: float
    timeout: timedelta = timedelta(minutes=config.MAX_WAITING_TIME)  # 请求超时时间
    fcs_reach_cost: float = 0.0
