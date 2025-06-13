import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import traceback
import sklearn
from sklearn import decomposition, preprocessing, cluster, metrics, model_selection
from langchain.tools import Tool
from langchain.chains import LLMChain
from utils.code_utils import clean_code_for_streamlit, extract_code_from_response, extract_explanation_from_response
from prompts.templates import viz_prompt

def create_viz_tool(llm):
    """
    Creates and returns a visualization tool that can generate and execute plotly code for data visualization.
    """
    viz_chain = LLMChain(llm=llm, prompt=viz_prompt)

    def viz_tool_func(input_str: str) -> str:
        """
        Accepts a single string input (the user's visualization request), and runs the Viz chain.
        """
        df = st.session_state.data.copy()
        columns_str = ", ".join(df.columns.tolist())
        request = input_str
        llm_response = viz_chain.run(columns=columns_str, request=request)
        st.write("#### [Viz Tool] LLM Raw Response:")
        st.write(llm_response)
        code_part = extract_code_from_response(llm_response)
        explanation_part = extract_explanation_from_response(llm_response)
        exec_namespace = {
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
        cleaned = clean_code_for_streamlit(code_part).strip()
        if "st.plotly_chart" not in cleaned and "fig" in cleaned:
            cleaned += "\nst.plotly_chart(fig)"
        try:
            exec(cleaned, {}, exec_namespace)
        except Exception as err:
            st.error(f"An error occurred while executing the generated code for the Viz tool.")
            st.code(cleaned, language="python")
            tb = traceback.format_exc()
            return f"Error executing Viz code:\n```\n{tb}\n```"
        return f"result: (see Streamlit chart)\nExplanation: {explanation_part}"

    return Tool(
        name="Visualization Tool",
        func=viz_tool_func,
        description="Use this to create visualizations of the dataset. Input should be a visualization request string (e.g., 'Show a scatter plot of A vs B')."
    ) 