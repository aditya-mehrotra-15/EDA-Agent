import streamlit as st
import pandas as pd
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler

from config.setup import setup_environment
from tools.eda_tool import create_eda_tool
from tools.viz_tool import create_viz_tool

# Initialize session state
if "data" not in st.session_state:
    st.session_state.data = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# Setup environment and get LLM
llm = setup_environment()

# Create tools
tools = [
    create_eda_tool(llm),
    create_viz_tool(llm)
]

# Initialize agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3,
    early_stopping_method="generate",
)

# Streamlit UI
st.title("EDA Agent")
st.write("Upload a CSV file to begin")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    if st.session_state.file_name != uploaded_file.name:
        st.session_state.file_name = uploaded_file.name
        st.session_state.data = pd.read_csv(uploaded_file)
        st.write("#### Dataset Preview:")
        st.dataframe(st.session_state.data.head())
        st.write("#### Dataset Info:")
        st.write(f"Shape: {st.session_state.data.shape}")
        st.write("Columns:", ", ".join(st.session_state.data.columns.tolist()))

if st.session_state.data is not None:
    user_input = st.text_input("What would you like to know about your data?")
    if user_input:
        with st.spinner("Thinking..."):
            response = agent.run(
                user_input,
                callbacks=[StreamlitCallbackHandler(st.container())]
            )
            st.write("#### Agent's Response:")
            st.write(response)
