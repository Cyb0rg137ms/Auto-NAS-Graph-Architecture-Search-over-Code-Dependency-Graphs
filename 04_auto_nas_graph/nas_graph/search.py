"""
search.py
=========
Search algorithm for Automated Neural Architecture Search (Auto-NAS).
Computes optimal layer selections based on target latency constraints and 
codebase complexity profiles.
"""

from typing import List, Dict, Any

CANDIDATE_LAYERS = [
    {"name": "LinearProjection", "flops_factor": 1.0, "param_factor": 1.2, "latency_ms_factor": 0.1},
    {"name": "ConvolutionalBlock", "flops_factor": 8.0, "param_factor": 4.5, "latency_ms_factor": 0.8},
    {"name": "AttentionBlock", "flops_factor": 15.0, "param_factor": 8.0, "latency_ms_factor": 1.5},
    {"name": "ResidualBlock", "flops_factor": 9.5, "param_factor": 5.0, "latency_ms_factor": 0.9},
    {"name": "LayerNorm", "flops_factor": 0.2, "param_factor": 0.1, "latency_ms_factor": 0.05}
]

class ArchitectureSearchEngine:
    """Orchestrates neural architecture selection based on complexity profiles and constraints."""
    
    def __init__(self, codebase_metrics: Dict[str, Any]):
        """
        Args:
            codebase_metrics: Summary metrics derived from the AST parser.
        """
        self.metrics = codebase_metrics

    def search(self, max_latency_ms: float, max_params_million: float) -> Dict[str, Any]:
        """
        Searches for the optimal configuration of layers that satisfies constraints
        and matches the codebase profile.
        
        Args:
            max_latency_ms: Hard constraint for model latency.
            max_params_million: Hard constraint for total parameter scale.
            
        Returns:
            A dictionary defining the selected layer stack and diagnostic scores.
        """
        # Determine base model depth from complexity metric
        base_depth = int(max(2, min(24, self.metrics["mean_complexity"] // 2)))
        
        selected_layers = []
        current_latency = 0.0
        current_params = 0.0
        
        # Iteratively build the layer configuration
        for i in range(base_depth):
            # Select candidate layer based on index and constraints
            # Alternate layers: project -> convolve/attention -> normalize -> residual
            if i == 0 or i == base_depth - 1:
                candidate = CANDIDATE_LAYERS[0]  # Linear Projection at boundaries
            elif i % 3 == 0:
                candidate = CANDIDATE_LAYERS[2]  # Attention Block
            elif i % 2 == 0:
                candidate = CANDIDATE_LAYERS[3]  # Residual Block
            else:
                candidate = CANDIDATE_LAYERS[1]  # Convolutional Block
                
            # Check if candidate fits in remaining budget
            if (current_latency + candidate["latency_ms_factor"] <= max_latency_ms and 
                current_params + candidate["param_factor"] * 0.1 <= max_params_million):
                
                selected_layers.append(candidate["name"])
                current_latency += candidate["latency_ms_factor"]
                current_params += candidate["param_factor"] * 0.1
            else:
                # Add a lightweight LayerNorm instead
                lightweight = CANDIDATE_LAYERS[4]
                if (current_latency + lightweight["latency_ms_factor"] <= max_latency_ms and 
                    current_params + lightweight["param_factor"] * 0.1 <= max_params_million):
                    selected_layers.append(lightweight["name"])
                    current_latency += lightweight["latency_ms_factor"]
                    current_params += lightweight["param_factor"] * 0.1
                    
        return {
            "network_depth": len(selected_layers),
            "layers": selected_layers,
            "estimated_latency_ms": current_latency,
            "estimated_params_million": current_params,
            "codebase_complexity_alignment": self.metrics["mean_complexity"]
        }
