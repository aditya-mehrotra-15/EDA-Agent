import streamlit as st
import pandas as pd
import traceback
import sklearn
from sklearn import decomposition, preprocessing, cluster, metrics, model_selection
from langchain.tools import Tool
from langchain.chains import LLMChain
from utils.code_utils import clean_code_for_streamlit, extract_code_from_response, extract_explanation_from_response
from prompts.templates import eda_prompt

def create_eda_tool(llm):
    """
    Creates and returns an EDA tool that can generate and execute pandas code for data analysis.
    """
    eda_chain = LLMChain(llm=llm, prompt=eda_prompt)

    def eda_tool_func(input_str: str) -> str:
        """
        Accepts a single string input (the user's question), and runs the EDA chain.
        """
        df = st.session_state.data.copy()
        columns_str = ", ".join(df.columns.tolist())
        question = input_str
        llm_response = eda_chain.run(columns=columns_str, question=question)
        st.write("#### [EDA Tool] LLM Raw Response:")
        st.write(llm_response)
        code_part = extract_code_from_response(llm_response)
        explanation_part = extract_explanation_from_response(llm_response)
        if not explanation_part:
            explanation_part = "The code was executed to generate the output seen in the Streamlit app."
        exec_namespace = {
            "pd": pd, 
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
        is_bare_expr = ("=" not in cleaned) and ("st.write" not in cleaned) and bool(cleaned)
        if is_bare_expr:
            cleaned = f"result = {cleaned}\nst.write(result)"
        try:
            exec(cleaned, {}, exec_namespace)
        except Exception as err:
            st.error(f"An error occurred while executing the generated code for the EDA tool.")
            st.code(cleaned, language="python")
            tb = traceback.format_exc()
            return f"Error executing EDA code:\n```\n{tb}\n```"
        result_val = exec_namespace.get("result", None)
        if result_val is not None:
            return f"result: {result_val}\nExplanation: {explanation_part}"
        else:
            return f"result: (see Streamlit output for details)\nExplanation: {explanation_part}"

    return Tool(
        name="EDA Tool",
        func=eda_tool_func,
        description="Use this to answer questions or perform analysis on the dataset. Input should be a question string (e.g., 'What is the mean of column A?')."
    ) 