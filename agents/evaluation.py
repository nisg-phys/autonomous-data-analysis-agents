from typing import Dict, List

class EvaluationAgent:
    """Evaluates the quality and correctness of the autonomous data analysis pipeline"""

    def __init__(self):
        pass
    def run(self, state: Dict)-> Dict:
        """Evaluate system outputs and returns validation metrics and verdict"""
        issues=[]

        data_score= self.evaluate_data_quality(state, issues)
        model_score= self.evaluate_model_quality(state,issues)

        system_score=round(0.4*data_score+0.6*model_score, 3)
        verdict= "Pass" if system_score>=0.7 else "Fail"
        return{
            "data_quality_score":data_score,
            "model_quality_score":model_score,
            "system_score": system_score,
            "issues": issues,
            "verdict": verdict
        }
    def evaluate_data_quality(self, state:Dict, issues: List[str])-> float:
        eda = state["eda_summary"]
        missing_values= eda.get("missing_values",{})
        total_missing= sum(missing_values.values())

        if total_missing > 0:
            issues.append(f" Dataset contains {total_missing} missing values.")

        imbalance= eda.get("class_distribution",None)    

        if imbalance:
            max_ratio = max(imbalance.values())/sum(imbalance.values())
            if max_ratio > 0.85:
                issues.append("Severe class imbalance detected")
        score= 1.0

        if total_missing>0:
            score-= 0.3
        if imbalance and max_ratio > 0.85:
            score -= 0.3
        return max(0.0, round(score,3))     
               
    def evaluate_model_quality(self, state: Dict, issues: List[str])-> float :
        metrics= state["model_metrics"]
        if "accuracy" in metrics:
            acc= metrics["accuracy"]
            if acc< 0.7:
                issues.append(f"Low classification accuracy:{acc}")
            return round(min(1.0, acc),3)    
        if "rmse" in metrics:
            rmse= metrics["rmse"]
            if rmse>10:
                issues.append(f"High regression RMSE: {rmse}")
            return round(max(0.0,1-rmse/100),3)
        issues.append("Unknow evaluation metric.")    
        return 0.0
