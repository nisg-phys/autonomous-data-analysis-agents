from typing import Dict
import json
import os
from langchain_groq import ChatGroq

class ExplanationAgent:
    """
    Uses an LLM to convert structured results into human-readable explanations
    """
    def __init__(self, model: str= "llama-3.1-8b-instant"):
        self.llm = ChatGroq(
             model=model,
             temperature=0.3
        )

    def run(self,
            eda_summary:Dict,
            model_metrics: Dict,
            task_type: str,
            target_column: str
            )-> str:

            prompt= self._build_prompt(
                 eda_summary, model_metrics, task_type, target_column
            )
            response = self.llm.invoke(prompt)
            return response.content
    def _build_prompt(
              self,
              eda_summary :Dict,
              model_metrics : Dict,
              task_type: str,
              target_column: str
    )-> str:
         return f"""
You are given the results of an automonous data analysis system. You are given the results of an autonomous data analysis system.
Task type:{task_type}
Target variable: {target_column}

Exploratory Data Analysis summary:
{json.dumps(eda_summary, indent=2)}

Model Evaluation metrics:
{json.dumps(model_metrics, indent=2)}

Write a concise professional report explaining:
1. Key characteristics of the dataset
2. Any data quality issues
3. Model performance interpretation
4. Practical implications of these results
"""