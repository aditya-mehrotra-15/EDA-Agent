import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
import sklearn
from sklearn import decomposition, preprocessing, cluster, metrics, model_selection
from langchain.tools import Tool
from utils.code_utils import extract_code

def create_python_exec_tool():
    """Creates a Python execution tool for arbitrary code."""
    def python_exec_tool_func(code: str) -> str:
        df = st.session_state.data.copy()
        st.subheader("[Python Exec Tool] Code Input")
        st.code(code, language="python")

        snippet = extract_code(code) or code
        # Execute snippet
        namespace = dict(pd=pd, px=px, go=go, df=df, st=st,
                         sklearn=sklearn, decomposition=decomposition,
                         preprocessing=preprocessing, cluster=cluster,
                         metrics=metrics, model_selection=model_selection)
        try:
            exec(snippet, {}, namespace)
        except Exception:
            st.error("Error executing Python code:")
            st.code(snippet, language="python")
            return f"Error: {traceback.format_exc()}"

        result = namespace.get("result")
        return f"result: {result}" if result is not None else "result: (see Streamlit output)"

    return Tool(
        name="Python Execution Tool",
        func=python_exec_tool_func,
        description="A Python REPL. Input: Python code snippet (e.g., 'df.shape')"
    )