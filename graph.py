from langgraph.graph import StateGraph, END
from state import AnalysisState
from agents.planner import PlannerAgent
from agents.eda import EDAAgent
from agents.modeling import ModelingAgent
from agents.explanation import ExplanationAgent
from agents.evaluation import EvaluationAgent

def planner_node(state: dict) -> dict:
    planner = PlannerAgent()
    result = planner.run(state["dataset_path"])

    return {**state,
        "dataset_summary": result["dataset_summary"],
        "task_type": result["task_type"],
        "target_column": result["target_column"],
        "plan": result["plan"]
    }

def eda_node(state: dict) -> dict:
    eda_agent = EDAAgent()
    result = eda_agent.run(
        dataset_path= state["dataset_path"],
        target_column= state["target_column"]
    )

    return {**state,
        "eda_summary": result["eda_summary"],
        "eda_plots_path": result["eda_plots_path"]
    }

def modeling_node(state: dict) -> dict:
    modeling_agent= ModelingAgent()
    result = modeling_agent.run(
        dataset_path= state["dataset_path"],
        target_column= state["target_column"],
        task_type=state["task_type"]
    )

    return {**state,
        "model_type": result["model_type"],
        "model_metrics": result["model_metrics"],
        "model_artifact_path":result["model_artifact_path"]
    }

def explanation_node(state:dict) -> dict:
    explanation_agent= ExplanationAgent()

    final_report= explanation_agent.run(
        eda_summary=state["eda_summary"],
        model_metrics= state["model_metrics"],
        task_type= state["task_type"],
        target_column=state["target_column"]
    )

    return {**state,
        "final_report": final_report
    }
def evaluation_node(state: dict)-> dict:
    evaluator=EvaluationAgent()
    evaluation= evaluator.run(state)
    return {
        **state,
        "evaluation": evaluation
    }
graph = StateGraph(dict)
graph.add_node("planner", planner_node)
graph.add_node("eda", eda_node)
graph.add_node("modeling",modeling_node)
graph.add_node("explanation", explanation_node)
graph.add_node("evaluation", evaluation_node)

graph.set_entry_point("planner")
graph.add_edge("planner", "eda")
graph.add_edge("eda", "modeling")
graph.add_edge("modeling", "explanation")
graph.add_edge("explanation", "evaluation")
graph.add_edge("evaluation", END)

app = graph.compile()


if __name__ == "__main__":
    initial_state = {
        "dataset_path": "data/adult.csv"}
   
    final_state = app.invoke(initial_state)

    from pprint import pprint

    print("\n=== FULL GRAPH EXECUTION COMPLETED ===")
    pprint(final_state["evaluation"])
    print(final_state["final_report"])
    #print(f"without pprint:{final_state}")
    #print("Target:", final_state["target_column"])
    # print("EDA keys:", final_state["eda_summary"]["missing_values"].values())
    # print("EDA keys:", final_state["eda_summary"].get("missing_values",{}).values())
    # print(final_state["eda_summary"].get("class_distribution", None ))
    # missing_values= eda.get("missing_values",{})
    #     total_missing= sum(missing_values.values())
 