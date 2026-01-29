from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class AnalysisState:
    #---Inputs-----
    dataset_path: str
    #----Planner outputs----
    dataset_summary: Optional[Dict[str,Any]]=None
    task_type: Optional[str]= None
    target_column: Optional[str]= None
    plan: Optional[Dict[str,Any]] = None
    {
        "steps": ["eda", "model", "evaluate"]
    }
    #---EDA outputs----
    eda_summary:Optional[Dict[str,Any]]=None
    eda_plots_path: Optional[str]= None

    #---Error Handling-----
    errors: Optional[str]= None
    model_type:Optional[str]=None
    model_metrics:Optional[Dict[str, Any]]= None
    model_artifact_path: Optional[str]= None

    #--------Explanation outputs-------
    final_report:Optional[str]= None