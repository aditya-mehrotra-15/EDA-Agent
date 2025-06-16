import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
import sklearn
from sklearn import decomposition, preprocessing, cluster, metrics, model_selection
from langchain.tools import Tool
from utils.code_utils import clean_code_for_streamlit

def create_python_exec_tool():
    """
    Creates and returns a Python Execution Tool (a "raw Python REPL") that executes arbitrary Python code.
    It accepts a single string (the raw Python code) and executes it in a namespace where df, pd, px, go, st, and sklearn (and its submodules) are available.
    It returns a string (e.g., "result: ...") or an error message.
    """
    def python_exec_tool_func(input_str: str) -> str:
        df = st.session_state.data.copy()
        namespace = {
            "pd": pd, 
            "px": px, 
            "go": go, 
            "df": df, 
            "st": st, 
            "sklearn": sklearn, 
            "decomposition": decomposition, 
            "preprocessing": preprocessing, 
            "cluster": cluster, 
            "metrics": metrics, 
            "model_selection": model_selection
        }
        cleaned = clean_code_for_streamlit(input_str).strip()
        is_bare_expr = ("=" not in cleaned) and ("st.write" not in cleaned) and bool(cleaned)
        if is_bare_expr:
            cleaned = f"result = {cleaned}\nst.write(result)"
        else:
            # If assignment but no st.write, append st.write on assigned var
            if ("=" in cleaned) and ("st.write" not in cleaned):
                first_line = cleaned.splitlines()[0]
                lhs = first_line.split("=", 1)[0].strip()
                if lhs.isidentifier():
                    cleaned += f"\nst.write({lhs})"
        try:
            exec(cleaned, {}, namespace)
        except Exception as err:
            st.error(f"An error occurred while executing the generated code for the Python Execution tool.")
            st.code(cleaned, language="python")
            tb = traceback.format_exc()
            return f"Error in Python Execution:\n```\n{tb}\n```"
        result_val = namespace.get("result", None)
        if result_val is not None:
            return f"result: {result_val}"
        else:
            return "result: (see Streamlit output)"
    return Tool(
        name="Python Execution Tool",
        func=python_exec_tool_func,
        description="A simple Python REPL. Input should be arbitrary Python code (e.g., 'df.shape', 'df.columns'). It executes the code and prints to the app, but does not return a summary of the analysis. For multi-step analysis or questions, use the EDA Tool."
    ) 