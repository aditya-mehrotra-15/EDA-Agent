import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
import sklearn
from sklearn import decomposition, preprocessing, cluster, metrics, model_selection
from langchain.tools import Tool
from langchain.chains import LLMChain
from utils.code_utils import extract_code, extract_explanation
from prompts.templates import viz_prompt

def create_viz_tool(llm):
    """Creates a visualization tool using Plotly and scikit-learn."""
    viz_chain = LLMChain(llm=llm, prompt=viz_prompt)

    def viz_tool_func(request: str) -> str:
        df = st.session_state.data.copy()
        cols = ", ".join(df.columns.tolist())

        # Generate LLM response
        llm_response = viz_chain.run(columns=cols, request=request)
        st.subheader("[Viz Tool] LLM Raw Response")
        st.markdown(llm_response)

        # Extract and show code snippet
        code_snippet = extract_code(llm_response)
        if code_snippet:
            st.code(code_snippet, language="python")

        explanation = extract_explanation(llm_response) or "Visualization created successfully."

        # Execute snippet
        namespace = dict(pd=pd, px=px, go=go, df=df, st=st,
                         sklearn=sklearn, decomposition=decomposition,
                         preprocessing=preprocessing, cluster=cluster,
                         metrics=metrics, model_selection=model_selection)
        try:
            exec(code_snippet, {}, namespace)
        except Exception:
            st.error("Error executing visualization code:")
            st.code(code_snippet, language="python")
            return f"Error: {traceback.format_exc()}"

        return f"result: (see Streamlit chart)\nExplanation: {explanation}"

    return Tool(
        name="Visualization Tool",
        func=viz_tool_func,
        description="Use for data visualization. Input: a request string, e.g., 'Show a scatter plot of A vs B'"
    )
