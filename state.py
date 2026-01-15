from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AnalysisState:
    dataset_path: str
    dataset_summary: Optional[Dict[str,Any]]=None
    task_type: Optional[str]= None
    target_column: Optional[str]= None
    plan: Optional[Dict[str,Any]] = None
    {
        "steps": ["eda", "model", "evaluate"]
    }
    errors: Optional[str]= None