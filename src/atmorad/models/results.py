from typing import Self

from pydantic import BaseModel, ConfigDict, Field


class BaseResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def merge(self, other: Self) -> Self:
        raise NotImplementedError


class EngineResult(BaseResult):
    cpu_time_s: float = Field(default=0.0, ge=0.0)
    simulation_time_s: float = Field(default=0.0, ge=0.0)

    def merge(self, other: Self) -> Self:
        return self.__class__(
            cpu_time_s=self.cpu_time_s + other.cpu_time_s,
            simulation_time_s=self.simulation_time_s + other.simulation_time_s,
        )


class SimulationResults(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    engine: EngineResult = EngineResult()
    detectors: dict[str, BaseResult] = {}

    def merge(self, other: Self) -> Self:
        merged_detectors = {}
        all_keys = set(self.detectors.keys()).union(other.detectors.keys())

        for key in all_keys:
            if key in self.detectors and key in other.detectors:
                merged_detectors[key] = self.detectors[key].merge(other.detectors[key])
            elif key in self.detectors:
                merged_detectors[key] = self.detectors[key]
            else:
                merged_detectors[key] = other.detectors[key]

        return self.__class__(engine=self.engine.merge(other.engine), detectors=merged_detectors)
