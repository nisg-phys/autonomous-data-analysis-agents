from agents.planner import PlannerAgent
from state import AnalysisState

def main():
    state= AnalysisState(dataset_path="data/raw/adult.csv")
    planner= PlannerAgent()
    result = planner.run(state.dataset_path)
    state.dataset_summary = result["dataset_summary"]
    state.plan = result["plan"]
    state.task_type = result["plan"]["task_type"]
    state.target_column = result["plan"]["target_column"]
    print("\n===PLANEER OUTPUT ==="")
    print(state.plan)
if __name__ == "__main__":
    main()
