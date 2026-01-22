from agents.planner import PlannerAgent
from agents.eda import EDAAgent
from state import AnalysisState
from agents.modeling import ModelingAgent

def main():
    state= AnalysisState(dataset_path="data/raw/adult.csv")
    planner= PlannerAgent()
    result = planner.run(state.dataset_path)
    state.dataset_summary = result["dataset_summary"]
    state.plan = result["plan"]
    state.task_type = result["plan"]["task_type"]
    state.target_column = result["plan"]["target_column"]
    #print("\n===PLANEER OUTPUT ===")
    #print(state.plan)
    #print(state.dataset_summary)

    #----EDA-----
    eda_agent = EDAAgent()
    eda_result = eda_agent.run(dataset_path= state.dataset_path,
                               target_column=state.target_column)
    state.eda_summary = eda_result["eda_summary"]
    state.eda_plots_path = eda_result["eda_plots_path"]

    print("\n=====EDA COMPLETE=====")
    print("EDA summary keys:", state.eda_summary.keys())
    print("Numeric summary keys:", state.eda_summary["target_distribution"].keys())
    print("EDA plots saved at:", state.eda_plots_path)

    #---------------Modeling----------------------

    modeling_agent= ModelingAgent()
    modeling_result = modeling_agent.run(
        dataset_path=state.dataset_path,
        target_column=state.target_column,
        task_type= state.task_type
    )
    state.model_type= modeling_result["model_type"]
    state.model_metrics = modeling_result["model_metrics"]
    state.model_artifact_path= modeling_result["model_artifact_path"]

    print("\n=== MODELING COMPLETED===")
    print("Model type:", state.model_type)
    print("Metrics", state.model_metrics)
    print("Model saved at", state.model_artifact_path)
if __name__ == "__main__":
    main()
