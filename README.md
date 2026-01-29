# Autonomous Multi-Agent Data Analysis System  
*(LangGraph + LLM + Evaluation)*

This repository implements an **autonomous data analysis pipeline** built using a **multi-agent architecture**. The system performs **end-to-end structured data analysis** that includes

- Exploratory data analysis (EDA)
- Machine learning model training using *scikit-learn*
- Natural-language explanation generation using a large language model (LLM)
- Automated system evaluation and validation

Two execution modes are supported:

1. **Sequential execution (`main.py`)**
2. **Graph-based orchestration using LangGraph (`graph.py`)**

The project is intended primarily as a **learning and research-oriented system** to explore:

- agent-based AI workflows  
- graph-based orchestration  
- structured AI system evaluation  

---

## What This System Does

Given a structured dataset (CSV file), the system automatically:

- Infers the machine learning task type (classification or regression)
- Performs statistical analysis and visualization
- Trains an appropriate machine learning model
- Generates a natural-language explanation of results
- Evaluates overall pipeline quality using explicit scoring criteria

---

## Key Concepts 

### **Multi-Agent System**

Instead of writing one large script, the pipeline is broken into **multiple specialized agents**, each responsible for a single task. This improves:

- modularity  
- clarity  
- debuggability  
- extensibility  

---

### **Orchestration**

Orchestration refers to **coordinating multiple agents** so they execute in the correct order, share information, and pass results between steps.

---

### **State**

A shared **state object (dictionary)** stores intermediate outputs produced by each agent and is passed across the pipeline.

---

### **LangGraph**

LangGraph is a framework for building **graph-based execution pipelines**, where each agent is treated as a node in a directed execution graph.

---

## System Architecture

### High-Level Execution Flow

# Autonomous Multi-Agent Data Analysis System  
*(LangGraph + LLM + Evaluation)*

This repository implements an **autonomous data analysis pipeline** built using a **multi-agent architecture**. The system performs **end-to-end structured data analysis**, including:

- Exploratory data analysis (EDA)
- Machine learning model training using *scikit-learn*
- Natural-language explanation generation using a large language model (LLM)
- Automated system evaluation and validation

Two execution modes are supported:

1. **Sequential execution (`main.py`)**
2. **Graph-based orchestration using LangGraph (`graph.py`)**

The project is intended primarily as a **learning and research-oriented system** to explore:

- agent-based AI workflows  
- graph-based orchestration  
- structured AI system evaluation  

---

## System Architecture



### Execution Graph

        ┌──────────────┐
        │   Planner    │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │     EDA      │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │  Modeling    │
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Explanation  │  ← LLM
        └──────┬───────┘
               ↓
        ┌──────────────┐
        │ Evaluation   │
        └──────┬───────┘
               ↓
              END

---

## Agent Responsibilities

| Agent | Description |
|---------|--------------|
| **Planner Agent** | Determines task type (classification / regression), target column, and analysis plan |
| **EDA Agent** | Computes descriptive statistics, distributions, and visualizations |
| **Modeling Agent** | Trains machine learning models and computes evaluation metrics |
| **Explanation Agent** | Uses an LLM to generate natural-language interpretation of results |
| **Evaluation Agent** | Audits data quality, model performance, and system reliability |

---
## Project Structure

autonomous-data-analysis-agents/
│
├── agents/
│ ├── planner.py
│ ├── eda.py
│ ├── modeling.py
│ ├── explanation.py
│ └── evaluation.py
│
├── data/
│ └── raw/
│ └── adult.csv
│
├── artifacts/
│ ├── plots/
│ └── models/
│
├── graph.py
├── main.py
├── state.py
├── test.py
├── requirements.txt
└── README.md


---

## File Descriptions

- **graph.py**  
  Implements **LangGraph-based execution**, treating each agent as a node in a directed execution graph.

- **main.py**  
  Implements a **traditional sequential pipeline**, executing agents step-by-step and explicitly managing shared state defined in `state.py`.

- **state.py**  
  Defines the shared data structure used to store intermediate results.

- **test.py**  
  Contains testing code for pandas-based data operations and debugging.

- **artifacts/**  
  Stores plots and trained models produced during execution.

---

## Execution Modes

### 1. Sequential Pipeline — `main.py`

This mode executes agents in a **fixed linear order**, manually passing state between agents.

**Flow:**

Planner → EDA → Modeling → Explanation → END


**Run:**

python main.py

**Use cases:**
Debugging
Learning agent interactions
Understanding pipeline logic

### 2. Graph-Based Pipeline — graph.py (LangGraph)
This mode uses LangGraph to orchestrate agents using an explicit execution graph.
Flow:
Planner → EDA → Modeling → Explanation → Evaluation → END
Run:
python graph.py


**Advantages:**
- Explicit execution control
- Better traceability
- Easier extensibility
- Support for conditional routing and retries

## Installation
### 1. Clone Repository

- git clone https://github.com/<your-username>/autonomous-data-analysis-agents.git
- cd autonomous-data-analysis-agents

### 2. Create Virtual Environment (Recommended)

- python -m venv venv
- source venv/bin/activate

### 3. Install Dependencies

pip install -r requirements.txt
pip install langgraph

## API Key Setup (LLM Integration)

This project uses **Groq LLM APIs** for explanation generation.

To set your API key, run the following command in your terminal:

bash
export GROQ_API_KEY="your_api_key_here"

- To persist this permanently, add the above line to one of the following files:
- macOS → ~/.zshrc
- Linux → ~/.bashrc
Then reload your shell configuration:

source ~/.zshrc

Verify that the key is set correctly:
echo $GROQ_API_KEY

## Dataset Setup

This project uses the **UCI Adult Income Dataset**.

Data set is downloaded through **download_data.py** and place it inside the `data/` directory:

- wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data -O data/raw/adult.csv

## Running the System
### Graph-Based Execution (Recommended)
Run the full autonomous pipeline using LangGraph:
- python graph.py

### Example Output (Abbreviated)
{
  "task_type": "classification",
  "target_column": "income",
  "model_metrics": {
      "accuracy": 0.86
  },
  "evaluation": {
      "data_quality_score": 0.95,
      "model_quality_score": 0.87,
      "system_score": 0.911,
      "verdict": "PASS"
  }
}
### Example LLM-Generated Explanation
The dataset contains demographic and employment-related attributes used
to predict income level. The model achieved 86% classification accuracy,
indicating strong predictive capability. Minimal missing values and low
class imbalance contributed to stable model performance.

**Evaluation Methodology**
The Evaluation Agent computes multiple quality metrics to validate the reliability of the pipeline.
**Data Quality Score**
- Missing value penalties
- Class imbalance detection
- Model Quality Score
- Classification accuracy thresholding
- Regression RMSE scoring
- System Score
The overall system score is computed as:
system_score = 0.4 × data_quality + 0.6 × model_quality
- Verdict
System Score	Verdict
≥ 0.70	PASS
< 0.70	FAIL

## Why Two Execution Modes?
Providing both procedural and graph-based pipelines allows:
- Easier learning and debugging (main.py)
- Advanced orchestration and extensibility (graph.py)
- This enables comparison between:
 Traditional machine learning pipelines
Agent-based orchestration systems
### Future Work
Planned extensions include:
- SQL-based autonomous analytics agent
- Financial reasoning multi-agent system
- Conditional routing and self-correction
- Multi-agent parallel execution
- Autonomous error recovery

# Notes
This project is designed primarily as:
An educational reference
A research-oriented system
An experimental platform for studying agentic AI workflows
It is not intended as a production deployment system.