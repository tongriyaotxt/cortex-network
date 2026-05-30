"""
Integrated Predictive Coding

Implements predictive coding principles inspired by:
- Friston, K. (2005). A free energy principle for the brain.
- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex.
- Bastos, A. M., et al. (2012). Canonical microcircuits for predictive coding.

Each layer predicts the activity of the layer above,
and prediction errors drive learning and perception.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PredictiveCodingLayer(nn.Module):
    """
    A layer that implements predictive coding dynamics.
    
    Given input h (from below), the layer:
    1. Predicts what the next layer should see: μ = f_pred(h)
    2. Receives actual next-layer activity h_next
    3. Computes precision-weighted prediction error: ε = π ⊙ (h_next - μ)
    4. Updates representations to minimize error
    """
    def __init__(self, d_model, d_pred=None, precision_dim=None, update_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_pred = d_pred or d_model
        self.update_rate = update_rate
        
        # Prediction function: what do we expect the next layer to look like?
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        
        # Precision (inverse variance) estimator
        # Models neuromodulatory control of precision (e.g., acetylcholine)
        precision_dim = precision_dim or d_model
        self.precision_estimator = nn.Sequential(
            nn.Linear(d_model, precision_dim),
            nn.SiLU(),
            nn.Linear(precision_dim, d_model),
            nn.Softplus(),  # Precision must be positive
        )
        
        # Error processing
        self.error_processor = nn.Linear(d_model, d_model)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, h_current, h_target=None, return_error=False):
        """
        Args:
            h_current: (batch, seq_len, d_model) current layer representation
            h_target: (batch, seq_len, d_pred) target representation to predict
                      If None, just computes prediction
            return_error: whether to return prediction error
        Returns:
            h_updated: updated representation
            prediction: predicted target
            error_info: dict with errors and precisions
        """
        # Generate prediction
        prediction = self.predictor(h_current)
        
        # Estimate precision (confidence in prediction)
        precision = self.precision_estimator(h_current) + 1e-4
        
        if h_target is None:
            # Just return prediction
            return prediction, {'precision': precision}
        
        # Compute prediction error
        prediction_error = h_target - prediction
        
        # Precision-weighted error
        weighted_error = precision * prediction_error
        
        # Update current representation (gradient descent on error)
        error_feedback = self.error_processor(weighted_error)
        h_updated = h_current + self.update_rate * error_feedback
        
        error_info = {
            'prediction': prediction,
            'error': prediction_error,
            'weighted_error': weighted_error,
            'precision': precision,
            'error_magnitude': prediction_error.norm(dim=-1).mean(),
        }
        
        if return_error:
            return h_updated, error_info
        return h_updated
    
    def compute_loss(self, h_current, h_target):
        """
        Compute prediction coding loss (free energy bound).
        
        L = E_q[log q(h|g) - log p(g,h)]
          ≈ ||ε||²_π + log det(π⁻¹)
        """
        prediction = self.predictor(h_current)
        precision = self.precision_estimator(h_current) + 1e-4
        
        error = h_target - prediction
        
        # Precision-weighted squared error
        weighted_error = precision * (error ** 2)
        reconstruction_loss = weighted_error.sum() / (h_current.size(0) * h_current.size(1))
        
        # Precision regularization (encourage confident predictions)
        # High precision = low variance = confident
        precision_reg = -torch.log(precision + 1e-6).mean()
        
        return reconstruction_loss, precision_reg


class HierarchicalPredictiveCoding(nn.Module):
    """
    Stack of predictive coding layers that form a hierarchy.
    
    Each layer tries to predict the one above it,
    and prediction errors propagate both up and down.
    """
    def __init__(self, d_model, n_layers=3, update_rate=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PredictiveCodingLayer(d_model, update_rate=update_rate)
            for _ in range(n_layers)
        ])
    
    def forward(self, representations):
        """
        Args:
            representations: list of tensors [h_0, h_1, ..., h_n]
                             where h_0 is bottom (input) and h_n is top
        Returns:
            updated_reps: updated representations
            total_error: sum of prediction errors
        """
        updated_reps = [r.clone() for r in representations]
        total_error = 0.0
        
        for i in range(len(self.layers)):
            if i + 1 < len(updated_reps):
                h_current = updated_reps[i]
                h_target = updated_reps[i + 1]
                
                h_updated, error_info = self.layers[i](
                    h_current, h_target, return_error=True
                )
                
                updated_reps[i] = h_updated
                total_error += error_info['error_magnitude']
        
        return updated_reps, total_error
    
    def compute_loss(self, representations):
        """Compute total predictive coding loss."""
        total_loss = 0.0
        total_precision_reg = 0.0
        
        for i in range(len(self.layers)):
            if i + 1 < len(representations):
                rec_loss, prec_reg = self.layers[i].compute_loss(
                    representations[i], representations[i + 1]
                )
                total_loss += rec_loss
                total_precision_reg += prec_reg
        
        return total_loss, total_precision_reg
