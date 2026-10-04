import json
import time
from typing import List
from abc import abstractmethod, ABCMeta


class Entity:
    @abstractmethod
    def to_dict(self):
        pass


class Header(Entity):
    def __init__(self, thing_id: str = '', key: str = '', channel: str = '', external_id: str = '',
                 external_key: str = ''):
        self.thing_id = thing_id
        self.key = key
        self.channel = channel
        self.external_id = external_id
        self.external_key = external_key

    def to_dict(self):
        return {
            "thing": self.thing_id,
            "channel": self.channel,
            "key": self.key,
            'external_id': self.external_id,
            'external_key': self.external_key
        }


class Payload(Entity):
    def __init__(self,
                 senml: SenML,
                 t: int = 1,
                 axis: int = 0,
                 frequency: int = 0,
                 action: int = 1,
                 length: int = 0,
                 coefficient: int = 0,
                 repair_factor: int = 0,
                 ):
        self.t = t
        self.axis = axis
        self.frequency = frequency
        self.action = action
        self.length = length
        self.coefficient = coefficient
        self.repair_factor = repair_factor
        self.senml = senml

    def to_dict(self):
        senml = {
            "n": self.senml.n,
            "bt": self.senml.bt,
            "u": self.senml.u,
            "vs": json.dumps(self.senml.vs.to_dict())
        }

        return {
            "type": self.t,
            "axis": self.axis,
            "frequency": self.frequency,
            'action': self.action,
            'length': self.length,
            'coefficient': self.coefficient,
            'repair_factor': self.repair_factor,
            'senml': senml,
        }


class Vib(Entity):
    def __init__(self, name: str = 'acce', sampling_rate: int = 10240, y_value: List[float] = None):
        self.name = name
        self.sampling_rate = sampling_rate
        self.y_value = y_value

    def to_dict(self):
        return {
            "name": self.name,
            "samplingRate": self.sampling_rate,
            "yValue": self.y_value
        }

    def to_json(self):
        return json.dumps(self.to_dict())


class SenML(Entity):
    def __init__(self, n: str = 'VIB1Z_H_W', bt: int = int(time.time()), u: str = 'm/s2', vs: str = []):
        self.n = n
        self.bt = bt
        self.u = u
        self.vs = vs

    def to_dict(self):
        return {
            "n": self.n,
            "bt": self.bt,
            "u": self.u,
            "vs": json.dumps(self.vs.to_dict())
        }


class Data(Entity):
    def __init__(self, header: Header, payload: List[Payload]):
        self.header = header
        self.payload = payload

    def to_dict(self):
        return {
            **self.header.to_dict(),
            "payload": [p.to_dict() for p in self.payload],
        }
