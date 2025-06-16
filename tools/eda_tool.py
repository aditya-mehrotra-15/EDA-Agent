import streamlit as st
import pandas as pd
import traceback
import sklearn
from sklearn import decomposition, preprocessing, cluster, metrics, model_selection
from langchain.tools import Tool
from langchain.chains import LLMChain
from utils.code_utils import extract_code, extract_explanation
from prompts.templates import eda_prompt

def create_eda_tool(llm):
    """Creates an EDA tool for data analysis with pandas and scikit-learn."""
    eda_chain = LLMChain(llm=llm, prompt=eda_prompt)

    def eda_tool_func(question: str) -> str:
        df = st.session_state.data.copy()
        cols = ", ".join(df.columns.tolist())

        # Generate LLM response
        llm_response = eda_chain.run(columns=cols, question=question)
        st.subheader("[EDA Tool] LLM Raw Response")
        st.markdown(llm_response)

        # Extract and show code snippet
        code_snippet = extract_code(llm_response)
        if code_snippet:
            st.code(code_snippet, language="python")

        explanation = extract_explanation(llm_response) or "Analysis executed successfully."

        # Execute snippet
        namespace = dict(pd=pd, df=df, st=st,
                         sklearn=sklearn, decomposition=decomposition,
                         preprocessing=preprocessing, cluster=cluster,
                         metrics=metrics, model_selection=model_selection)
        try:
            exec(code_snippet, {}, namespace)
        except Exception:
            st.error("Error executing EDA code:")
            st.code(code_snippet, language="python")
            return f"Error: {traceback.format_exc()}"

        result = namespace.get("result")
        return (f"result: {result}\nExplanation: {explanation}" 
                if result is not None 
                else f"result: (see Streamlit output)\nExplanation: {explanation}")

    return Tool(
        name="EDA Tool",
        func=eda_tool_func,
        description="Use for data analysis. Input: question string, e.g., 'What is the mean of column A?'"
    )
