"""
CHAPPIE Phase 6: Consciousness Transfer

Mind state serialization, backup, and cross-body transfer.
Like Chappie uploading his consciousness to a new robot body.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import json
import hashlib
from datetime import datetime
import math


class MindStateSerializer:
    """
    Serialize entire cognitive state into a portable format.
    
    Includes:
        - All neural weights (DCU branches, GNW projections)
        - All state vectors (multi-timescale, workspace, consciousness)
        - All memories (consolidated engrams, emotional triggers)
        - Personality parameters (emotion baseline, moral stage)
        - Self-narrative (life events, relationships, identity statements)
        - Trust model (individual trust scores, deception records)
    
    Output: a "consciousness file" that can be stored, transmitted, and loaded.
    """
    
    def __init__(self, compression: bool = True):
        self.compression = compression
        
    def serialize(
        self,
        model: nn.Module,
        emotion_system: Any,
        moral_module: Any,
        narrative_generator: Any,
        consolidator: Any,
        trust_model: Any,
        metadata: Optional[Dict] = None,
    ) -> Dict:
        """
        Serialize complete mind state.
        
        Returns:
            Dict with 'weights', 'states', 'memories', 'personality', 'narrative', 'trust', 'metadata'
        """
        state = {
            "version": "chappie-v1.0",
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        }
        
        # 1. Neural weights
        state["weights"] = {}
        for name, param in model.named_parameters():
            state["weights"][name] = {
                "shape": list(param.shape),
                "data": param.data.cpu().tolist() if param.numel() < 10000 else "<compressed>",
            }
        
        # 2. Emotional state
        if emotion_system is not None:
            state["emotion"] = {
                "current_state": emotion_system.state.to_vector().tolist(),
                "triggers": {k: v[0].tolist() for k, v in emotion_system.emotional_memory.items()},
                "target_valence": emotion_system.target_valence.item(),
                "target_arousal": emotion_system.target_arousal.item(),
            }
        
        # 3. Moral development
        if moral_module is not None:
            state["moral"] = {
                "stage": moral_module.stage,
                "rules": [
                    {"condition": r.condition, "action": r.action, "priority": r.priority,
                     "source": r.source, "stage": r.stage}
                    for r in moral_module.rules
                ],
                "experiences": moral_module.experiences[-20:],  # Recent experiences
            }
        
        # 4. Self-narrative
        if narrative_generator is not None:
            state["narrative"] = {
                "name": narrative_generator.name,
                "birth_context": narrative_generator.birth_context,
                "events": [
                    {"timestamp": e.timestamp, "description": e.description,
                     "location": e.location, "people": e.people,
                     "significance": e.significance, "tags": e.tags}
                    for e in narrative_generator.events
                ],
                "relationships": narrative_generator.relationships,
                "identity_statements": narrative_generator.identity_statements,
                "chapters": narrative_generator.chapters,
            }
        
        # 5. Memory engrams
        if consolidator is not None:
            state["memories"] = {
                "consolidated": [
                    {"pattern": e.pattern.tolist(), "importance": e.importance,
                     "branch_id": e.branch_id, "retrieval_count": e.retrieval_count}
                    for e in consolidator.consolidated_engrams
                ],
                "pending": [
                    {"pattern": e.pattern.tolist(), "importance": e.importance}
                    for e in consolidator.pending_engrams
                ],
            }
        
        # 6. Trust model
        if trust_model is not None:
            state["trust"] = {
                "scores": trust_model.trust_scores,
                "suspicion": trust_model.suspicion_scores if hasattr(trust_model, 'suspicion_scores') else {},
            }
        
        # Compute hash for integrity verification
        state_hash = hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]
        state["integrity_hash"] = state_hash
        
        return state
    
    def deserialize(self, state: Dict, target_model: nn.Module) -> Dict:
        """
        Load mind state into a target model.
        
        Returns:
            Dict with loaded components and adaptation notes.
        """
        notes = []
        
        # Verify integrity
        stored_hash = state.get("integrity_hash", "")
        computed_hash = hashlib.sha256(json.dumps(state, sort_keys=True, default=str).encode()).hexdigest()[:16]
        if stored_hash != computed_hash:
            notes.append("WARNING: Integrity hash mismatch — state may be corrupted")
        
        # Load weights (if shapes match)
        if "weights" in state:
            loaded = 0
            mismatched = 0
            for name, w_data in state["weights"].items():
                try:
                    param = dict(target_model.named_parameters())[name]
                    if list(param.shape) == w_data["shape"]:
                        if w_data["data"] != "<compressed>":
                            param.data.copy_(torch.tensor(w_data["data"]))
                            loaded += 1
                    else:
                        mismatched += 1
                        notes.append(f"Shape mismatch for {name}: expected {list(param.shape)}, got {w_data['shape']}")
                except KeyError:
                    notes.append(f"Parameter {name} not found in target model")
            notes.append(f"Loaded {loaded} weights, {mismatched} mismatched")
        
        return {
            "success": True,
            "version": state.get("version"),
            "timestamp": state.get("timestamp"),
            "notes": notes,
            "components": list(state.keys()),
        }
    
    def estimate_size(self, state: Dict) -> Dict[str, int]:
        """Estimate storage size of each component."""
        sizes = {}
        for key, value in state.items():
            json_str = json.dumps(value, default=str)
            sizes[key] = len(json_str.encode('utf-8'))
        sizes["total"] = sum(sizes.values())
        return sizes


class ConsciousnessBackupManager:
    """
    Automatic backup and versioning of consciousness state.
    
    Like Chappie making regular backups so he never truly dies.
    """
    
    def __init__(self, max_backups: int = 10, auto_interval: int = 100):
        self.max_backups = max_backups
        self.auto_interval = auto_interval
        self.backups: List[Dict] = []
        self.interaction_count = 0
        
    def should_backup(self, trigger: str = "periodic") -> bool:
        """Determine if backup should occur."""
        if trigger == "danger":
            return True
        if trigger == "manual":
            return True
        if trigger == "periodic":
            self.interaction_count += 1
            return self.interaction_count % self.auto_interval == 0
        return False
    
    def create_backup(self, mind_state: Dict, trigger: str = "periodic") -> Dict:
        """Create a new backup with metadata."""
        backup = {
            "id": f"backup_{len(self.backups):04d}",
            "trigger": trigger,
            "timestamp": datetime.now().isoformat(),
            "mind_state": mind_state,
            "size_bytes": len(json.dumps(mind_state, default=str).encode('utf-8')),
        }
        self.backups.append(backup)
        
        # Keep only recent backups
        if len(self.backups) > self.max_backups:
            self.backups = self.backups[-self.max_backups:]
        
        return backup
    
    def restore_backup(self, backup_id: str) -> Optional[Dict]:
        """Retrieve a backup by ID."""
        for b in self.backups:
            if b["id"] == backup_id:
                return b["mind_state"]
        return None
    
    def get_backup_history(self) -> List[Dict]:
        """List all backups with metadata."""
        return [{"id": b["id"], "trigger": b["trigger"], "timestamp": b["timestamp"],
                 "size_bytes": b["size_bytes"]} for b in self.backups]
    
    def diff_backups(self, id_a: str, id_b: str) -> Dict:
        """Show differences between two backups."""
        a = self.restore_backup(id_a)
        b = self.restore_backup(id_b)
        
        if a is None or b is None:
            return {"error": "Backup not found"}
        
        differences = {}
        for key in set(a.keys()) | set(b.keys()):
            if key not in a:
                differences[key] = "added in B"
            elif key not in b:
                differences[key] = "removed in B"
            elif json.dumps(a[key], sort_keys=True, default=str) != json.dumps(b[key], sort_keys=True, default=str):
                differences[key] = "modified"
        
        return {
            "backup_a": id_a,
            "backup_b": id_b,
            "differences": differences,
        }


class BodyAdapter:
    """
    Adapt mind state to a new body with different capabilities.
    
    When Chappie transfers to a new robot body, he needs to:
        - Map old motor primitives to new actuators
        - Adapt sensory resolution differences
        - Handle missing or additional sensors
    """
    
    def __init__(self):
        self.capability_mappings: Dict[str, Dict] = {}
        
    def analyze_body_capabilities(self, body_spec: Dict) -> Dict:
        """
        Analyze a target body's capabilities.
        
        body_spec: {
            "n_actuators": 20,
            "sensors": ["vision", "audio", "tactile", "proprioception"],
            "max_speed": 5.0,
            "sensor_resolution": {"vision": 1080, "audio": 44100},
        }
        """
        return {
            "motor_dof": body_spec.get("n_actuators", 0),
            "sensor_count": len(body_spec.get("sensors", [])),
            "capabilities": body_spec.get("sensors", []),
            "limitations": [],
        }
    
    def adapt_motor_primitives(
        self,
        old_primitives: Dict,
        old_dof: int,
        new_dof: int,
    ) -> Dict:
        """
        Map motor primitives from old body to new body.
        
        If new body has fewer DOF: merge/control reduction
        If new body has more DOF: expand with zeros/default
        """
        adapted = {}
        for name, primitive in old_primitives.items():
            if new_dof == old_dof:
                adapted[name] = primitive
            elif new_dof < old_dof:
                # Subsample: use every Nth actuator
                step = old_dof // new_dof
                adapted[name] = primitive[:, ::step]
            else:
                # Pad with zeros
                pad = torch.zeros(primitive.shape[0], new_dof - old_dof)
                adapted[name] = torch.cat([primitive, pad], dim=-1)
        
        return adapted
    
    def generate_adaptation_report(self, old_spec: Dict, new_spec: Dict) -> List[str]:
        """Generate human-readable adaptation notes."""
        notes = []
        
        old_sensors = set(old_spec.get("sensors", []))
        new_sensors = set(new_spec.get("sensors", []))
        
        lost = old_sensors - new_sensors
        gained = new_sensors - old_sensors
        
        if lost:
            notes.append(f"WARNING: Lost sensors: {lost}. Functionality may be impaired.")
        if gained:
            notes.append(f"INFO: New sensors available: {gained}. Enhanced perception possible.")
        
        old_dof = old_spec.get("n_actuators", 0)
        new_dof = new_spec.get("n_actuators", 0)
        
        if new_dof < old_dof:
            notes.append(f"NOTE: Reduced motor DOF from {old_dof} to {new_dof}. Some movements simplified.")
        elif new_dof > old_dof:
            notes.append(f"NOTE: Increased motor DOF from {old_dof} to {new_dof}. New movements possible.")
        
        return notes
