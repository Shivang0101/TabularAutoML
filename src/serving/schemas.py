# ─────────────────────────────────────────────────────────
# schemas.py — Pydantic Request and Response Schemas
# PURPOSE: Define the structure of all API inputs and outputs.
#          Pydantic validates data automatically — bad inputs return
#          a clear 422 Unprocessable Entity error with explanation.
# ─────────────────────────────────────────────────────────
 
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
 
 
class PredictRequest(BaseModel): 
    """
    Structure of /predict POST request body.
    Example JSON: {"features": {"age": 35, "income": 50000, "credit_score": 720}}
    """
    # Dict[str, Any] means: dictionary with string keys and any type of value
    # This is flexible because different datasets have different feature names/types
    features: Dict[str, Any] = Field(
        ...,  # '...' means this field is required (no default)
        description='Dictionary of feature name → feature value',
        example={'age': 35, 'income': 50000, 'credit_score': 720}
    )
 
    class Config:
        # Allow extra fields without raising validation errors
        # (Model will only use features it was trained on)
        extra = 'allow'
 
 
class PredictResponse(BaseModel):
    """Structure of /predict response."""
    # The predicted class label (e.g., 0 or 1 for binary classification)
    predicted_class: int
 
    # Confidence score: probability of the predicted class
    probability: float = Field(ge=0.0, le=1.0)  # Must be between 0 and 1
 
    # Full probability distribution across all classes
    all_probabilities: Dict[str, float]
 
 
class TrainResponse(BaseModel):
    """Structure of /train response."""
    status: str
    best_model_name: str
    auc_roc: float = Field(ge=0.0, le=1.0)
    n_features_selected: int
    top_3_models: List[str]
