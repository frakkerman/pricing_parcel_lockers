from dataclasses import dataclass
from typing import List

@dataclass
class Location:
    x: float
    y: float
    id_num: int #id only used for loaded data
    time: int#arrival time
    def __getitem__(self, item):
        return getattr(self, item)

@dataclass
class ParcelPoint:
    location: Location
    remainingCapacity: int
    id_num: int

@dataclass
class ParcelPoints:
    parcelpoints: List[ParcelPoint]
    def __getitem__(self, item):
        return getattr(self, item)

@dataclass
class Vehicle:
    routePlan: []
    capacity: int
    id_num: int
    def __getitem__(self, item):
        return getattr(self, item)
    
@dataclass
class Fleet:
    fleet: List[Vehicle]
    def __getitem__(self, item):
        return getattr(self, item)
    
@dataclass
class Customer:
    home: Location
    incentiveSensitivity: float
    home_util: float
    service_time: float
    id_num: int