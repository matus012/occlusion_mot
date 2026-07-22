"""CARLA occlusion-scenario feeder (P2, gate G4).

Named `sim` (not `carla`) to avoid colliding with the CARLA client package namespace.
The feeder renders scenarios into MOT17-style sequence dirs so every existing loader,
tracker, and eval path works on simulated data unchanged.
"""

from omot.sim.feeder import MockBackend, render_scenario
from omot.sim.scenario import OcclusionScenario, WalkerSpec, generate_scenarios

__all__ = [
    "MockBackend",
    "render_scenario",
    "OcclusionScenario",
    "WalkerSpec",
    "generate_scenarios",
]
