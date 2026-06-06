"""
CHAPPIE Phase 7b: Hardware Introspection

Body schema, damage detection, resource monitoring, self-repair planning.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math


class BodySchema:
    """
    Internal model of its own physical structure.
    
    Like Chappie knowing where his battery is, how many arms he has,
    and what happens when a motor fails.
    """
    
    def __init__(self, n_joints: int = 20, n_sensors: int = 10):
        self.n_joints = n_joints
        self.n_sensors = n_sensors
        
        # Body part registry
        self.parts: Dict[str, Dict] = {
            "head": {"type": "sensor_platform", "sensors": ["camera", "microphone", "speaker"], "actuators": ["neck_pitch", "neck_yaw"], "critical": True},
            "torso": {"type": "core", "sensors": ["battery_monitor", "temperature"], "actuators": ["waist_rotate"], "critical": True},
            "left_arm": {"type": "limb", "sensors": ["touch"], "actuators": ["shoulder", "elbow", "wrist", "gripper"], "critical": False},
            "right_arm": {"type": "limb", "sensors": ["touch"], "actuators": ["shoulder", "elbow", "wrist", "gripper"], "critical": False},
            "left_leg": {"type": "limb", "sensors": ["touch", "balance"], "actuators": ["hip", "knee", "ankle"], "critical": True},
            "right_leg": {"type": "limb", "sensors": ["touch", "balance"], "actuators": ["hip", "knee", "ankle"], "critical": True},
        }
        
        # Joint limits (min, max) in degrees
        self.joint_limits: Dict[str, Tuple[float, float]] = {}
        for part_name, part in self.parts.items():
            for actuator in part.get("actuators", []):
                self.joint_limits[f"{part_name}_{actuator}"] = (-180.0, 180.0)
        
    def get_part_info(self, part_name: str) -> Optional[Dict]:
        return self.parts.get(part_name)
    
    def is_critical(self, part_name: str) -> bool:
        part = self.parts.get(part_name)
        return part["critical"] if part else False
    
    def get_sensor_locations(self) -> Dict[str, str]:
        """Map each sensor to its body location."""
        locations = {}
        for part_name, part in self.parts.items():
            for sensor in part.get("sensors", []):
                locations[sensor] = part_name
        return locations
    
    def check_joint_safety(self, joint_name: str, target_angle: float) -> Tuple[bool, float]:
        """Check if a target angle is within safe limits."""
        limits = self.joint_limits.get(joint_name, (-180, 180))
        safe = limits[0] <= target_angle <= limits[1]
        margin = min(target_angle - limits[0], limits[1] - target_angle)
        return safe, margin


class DamageDetector:
    """
    Detect hardware damage from sensor readings and behavior anomalies.
    
    Chappie knows when his arm is broken because:
        - The motor doesn't respond
        - Joint angles don't match commands
        - Unusual temperature or vibration
    """
    
    def __init__(self, body_schema: BodySchema):
        self.body_schema = body_schema
        self.damage_log: List[Dict] = []
        
    def check_motor_health(self, joint_name: str, commanded: float, actual: float, current_draw: float) -> Dict:
        """
        Check motor health based on command-response mismatch.
        
        Returns:
            health_score: 0-1
            damage_type: description
        """
        error = abs(commanded - actual)
        
        if error > 30.0:  # Large tracking error
            return {"health": 0.2, "damage": " Severe tracking failure — possible mechanical damage", "joint": joint_name}
        elif error > 10.0:
            return {"health": 0.5, "damage": "Moderate tracking error — possible joint wear", "joint": joint_name}
        elif current_draw > 5.0:  # High current
            return {"health": 0.6, "damage": "High current draw — possible obstruction", "joint": joint_name}
        
        return {"health": 1.0, "damage": "None", "joint": joint_name}
    
    def check_sensor_health(self, sensor_name: str, reading: float, expected_range: Tuple[float, float]) -> Dict:
        """Check if a sensor is providing reasonable readings."""
        if reading < expected_range[0] or reading > expected_range[1]:
            return {"health": 0.3, "damage": f"Sensor {sensor_name} out of range", "sensor": sensor_name}
        
        # Check for stuck sensor (no variation)
        if hasattr(self, '_last_readings') and sensor_name in self._last_readings:
            if abs(reading - self._last_readings[sensor_name]) < 0.001:
                return {"health": 0.5, "damage": f"Sensor {sensor_name} appears stuck", "sensor": sensor_name}
        
        if not hasattr(self, '_last_readings'):
            self._last_readings = {}
        self._last_readings[sensor_name] = reading
        
        return {"health": 1.0, "damage": "None", "sensor": sensor_name}
    
    def full_system_check(self, joint_states: Dict, sensor_readings: Dict) -> Dict:
        """Run a comprehensive health check."""
        results = {
            "overall_health": 1.0,
            "damaged_parts": [],
            "warnings": [],
        }
        
        for joint_name, state in joint_states.items():
            health = self.check_motor_health(
                joint_name, state["commanded"], state["actual"], state["current"]
            )
            if health["health"] < 0.5:
                results["damaged_parts"].append(health)
                results["overall_health"] *= health["health"]
        
        for sensor_name, reading in sensor_readings.items():
            health = self.check_sensor_health(sensor_name, reading, (0, 100))
            if health["health"] < 0.5:
                results["warnings"].append(health)
        
        return results


class ResourceMonitor:
    """
    Track battery, compute, memory resources.
    
    Like Chappie feeling "hungry" when battery is low.
    """
    
    def __init__(self, battery_capacity: float = 100.0, compute_budget: float = 100.0):
        self.battery_capacity = battery_capacity
        self.compute_budget = compute_budget
        
        self.current_battery = battery_capacity
        self.current_compute = 0.0
        self.memory_usage = 0.0
        
        # Consumption rates per action type
        self.consumption_rates = {
            "idle": 0.1,
            "think": 1.0,        # CORTEX forward pass
            "move": 2.0,         # Motor control
            "perceive": 1.5,     # Vision/audio processing
            "communicate": 0.5,  # Speech synthesis
        }
        
    def consume(self, action_type: str, duration: float = 1.0):
        """Simulate resource consumption."""
        rate = self.consumption_rates.get(action_type, 1.0)
        self.current_battery -= rate * duration
        self.current_compute += rate * duration * 0.1
        self.current_battery = max(0, self.current_battery)
        self.current_compute = min(self.compute_budget, self.current_compute)
    
    def recharge(self, amount: float):
        """Recharge battery."""
        self.current_battery = min(self.battery_capacity, self.current_battery + amount)
    
    def get_battery_level(self) -> float:
        return self.current_battery / self.battery_capacity
    
    def get_battery_anxiety(self) -> float:
        """
        Emotional anxiety based on battery level.
        
        Like hunger — low battery triggers distress.
        """
        level = self.get_battery_level()
        if level < 0.1:
            return 1.0  # Panic
        elif level < 0.3:
            return 0.7  # High anxiety
        elif level < 0.5:
            return 0.3  # Mild concern
        return 0.0
    
    def predict_depletion(self, action_plan: List[str]) -> float:
        """
        Predict battery level after executing an action plan.
        
        Returns: estimated battery level
        """
        total_cost = sum(self.consumption_rates.get(a, 1.0) for a in action_plan)
        return max(0, self.current_battery - total_cost) / self.battery_capacity
    
    def should_seek_power(self) -> bool:
        """Determine if should actively seek charging."""
        return self.get_battery_level() < 0.2
    
    def optimize_power_mode(self, urgency: float = 0.5) -> Dict:
        """
        Suggest power optimization settings.
        
        Like going into "sleep mode" or reducing spike rates.
        """
        battery = self.get_battery_level()
        
        if battery < 0.1:
            return {
                "mode": "emergency",
                "cortex_layers_active": 2,
                "spike_threshold": 0.9,  # Very sparse firing
                "sensor_resolution": 0.25,
                "action": "FIND_CHARGER_IMMEDIATELY",
            }
        elif battery < 0.3:
            return {
                "mode": "conservative",
                "cortex_layers_active": 4,
                "spike_threshold": 0.7,
                "sensor_resolution": 0.5,
                "action": "REDUCE_NON_ESSENTIAL_ACTIVITY",
            }
        else:
            return {
                "mode": "normal",
                "cortex_layers_active": 8,
                "spike_threshold": 0.5,
                "sensor_resolution": 1.0,
                "action": "NORMAL_OPERATION",
            }
    
    def get_status(self) -> Dict:
        return {
            "battery_percent": self.get_battery_level() * 100,
            "compute_load": self.current_compute / self.compute_budget * 100,
            "battery_anxiety": self.get_battery_anxiety(),
            "needs_power": self.should_seek_power(),
        }


class SelfRepairPlanner:
    """
    Plan repair actions when damage is detected.
    
    Chappie can't fix everything himself, but he can:
        - Diagnose what's wrong
        - Determine if self-repair is possible
        - Find needed parts/tools
        - Plan repair sequence
    """
    
    def __init__(self, body_schema: BodySchema, damage_detector: DamageDetector):
        self.body_schema = body_schema
        self.damage_detector = damage_detector
        
    def plan_repair(self, damage_report: Dict) -> Optional[Dict]:
        """
        Generate a repair plan for detected damage.
        
        Returns:
            repair_plan or None if beyond self-repair
        """
        if not damage_report.get("damaged_parts"):
            return None
        
        plan = {
            "priority": "urgent",
            "steps": [],
            "can_self_repair": True,
            "needed_parts": [],
        }
        
        for damage in damage_report["damaged_parts"]:
            joint = damage["joint"]
            part_name = joint.split("_")[0] if "_" in joint else "unknown"
            part_info = self.body_schema.get_part_info(part_name)
            
            if part_info and part_info["critical"]:
                plan["priority"] = "critical"
            
            if "tracking failure" in damage["damage"]:
                plan["steps"].append(f"Reset motor controller for {joint}")
                plan["steps"].append(f"Check mechanical linkage on {joint}")
                plan["needed_parts"].append(f"{joint}_actuator")
            
            elif "wear" in damage["damage"]:
                plan["steps"].append(f"Apply lubrication to {joint}")
                plan["steps"].append(f"Recalibrate {joint} encoder")
            
            elif "obstruction" in damage["damage"]:
                plan["steps"].append(f"Clear obstruction from {joint}")
                plan["steps"].append(f"Test {joint} range of motion")
        
        # Determine if self-repair is possible
        if len(plan["needed_parts"]) > 2:
            plan["can_self_repair"] = False
            plan["steps"].append("SEEK_EXTERNAL_ASSISTANCE")
        
        return plan
    
    def assess_survivability(self, damage_report: Dict) -> float:
        """
        Assess if the body can still function with current damage.
        
        Returns: 0-1 survivability score
        """
        critical_damaged = 0
        total_critical = 0
        
        for part_name, part in self.body_schema.parts.items():
            if part["critical"]:
                total_critical += 1
                # Check if this part is damaged
                for damage in damage_report.get("damaged_parts", []):
                    if part_name in damage.get("joint", ""):
                        critical_damaged += 1
        
        if total_critical == 0:
            return 1.0
        
        return max(0, 1.0 - (critical_damaged / total_critical))
