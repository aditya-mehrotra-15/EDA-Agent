import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import json
import re

# Function to clean LLM-generated code for Streamlit

def clean_code_for_streamlit(code):
    # Replace print() with st.write()
    code = code.replace('print(', 'st.write(')
    # Replace xlabel/ylabel with xaxis_title/yaxis_title for Plotly
    code = re.sub(r'xlabel\s*=', 'xaxis_title=', code)
    code = re.sub(r'ylabel\s*=', 'yaxis_title=', code)
    # Remove lines with pd.read_csv, comments, markdown, or empty
    lines = code.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if (
            stripped.startswith('#')
            or stripped.startswith('```')
            or 'pd.read_csv' in stripped
            or stripped == ''
        ):
            continue
        if stripped.startswith('df ='):
            cleaned_lines.append('df = st.session_state.data')
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def extract_code_from_response(response):
    # Try to find a code block first
    match = re.search(r'```(?:python)?(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try to find 'Code:' (case-insensitive)
    match = re.search(r'Code:(.*)', response, re.IGNORECASE | re.DOTALL)
    if match:
        code = match.group(1)
        # If there's an Explanation, split it out
        code = code.split('Explanation:')[0]
        return code.strip()
    # Fallback: try to extract the first code-like lines (skip markdown/tables)
    lines = response.splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        if in_code or (line.strip() and not line.strip().startswith("|") and not line.strip().startswith("Output:") and not line.strip().startswith("Dataset:")):
            code_lines.append(line)
    return "\n".join(code_lines).strip()

def extract_explanation_from_response(response):
    # Try to find 'Explanation:' (case-insensitive)
    match = re.search(r'Explanation:(.*)', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

# Load environment variables
load_dotenv()

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'file_name' not in st.session_state:
    st.session_state.file_name = None

# Initialize Hugging Face model
repo_id = "HuggingFaceH4/zephyr-7b-beta"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    temperature=0.7,
    max_length=500,
    huggingfacehub_api_token=os.getenv('HUGGINGFACEHUB_ACCESS_TOKEN')
)

# Create prompt templates
eda_template = """You are a data analysis expert. Based on the following question about a dataset, provide Python code using pandas to answer the question and explain the results.
Dataset columns: {columns}
Question: {question}
Please provide the code and explanation in the following format ONLY:
Code: <code here>
Explanation: <explanation here>"""

eda_prompt = PromptTemplate(
    input_variables=["columns", "question"],
    template=eda_template
)

visualization_template = """You are a data visualization expert. Based on the following request, provide Python code using plotly to create the visualization.
Dataset columns: {columns}
Request: {request}
Please provide the code in the following format ONLY:
Code: <code here>"""

viz_prompt = PromptTemplate(
    input_variables=["columns", "request"],
    template=visualization_template
)

# Create chains
eda_chain = LLMChain(llm=llm, prompt=eda_prompt)
viz_chain = LLMChain(llm=llm, prompt=viz_prompt)

# Set up the Streamlit interface
st.title("📊 EDA Agent")
st.write("Upload your dataset and ask questions about it!")

# File upload section
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None and (st.session_state.file_name != uploaded_file.name):
    st.session_state.data = pd.read_csv(uploaded_file)
    st.session_state.file_name = uploaded_file.name
    st.success("File uploaded successfully!")

# Display dataset info if data is loaded
if st.session_state.data is not None:
    st.write("### Dataset Preview")
    st.dataframe(st.session_state.data.head())
    
    # Create tabs for EDA and Visualization
    tab1, tab2 = st.tabs(["📈 EDA Analysis", "🎨 Visualization"])
    
    with tab1:
        st.write("### Ask questions about your data")
        eda_question = st.text_input("Enter your question (e.g., 'What is the average of column X?' or 'Are there any missing values?')")
        
        if st.button("Analyze", key="eda_button"):
            if eda_question:
                with st.spinner("Analyzing..."):
                    # Get columns as string
                    columns_str = ", ".join(st.session_state.data.columns.tolist())
                    
                    # Get response from LLM
                    response = eda_chain.run(columns=columns_str, question=eda_question)
                    
                    # Extract code and explanation
                    try:
                        code_part = extract_code_from_response(response)
                        explanation_part = extract_explanation_from_response(response)
                        
                        # Display code
                        st.code(code_part, language="python")
                        
                        # Execute code and display results
                        try:
                            exec_locals = {"pd": pd, "df": st.session_state.data, "st": st}
                            code_to_run = clean_code_for_streamlit(code_part)
                            exec(code_to_run, {}, exec_locals)
                            
                            # Display explanation
                            st.write("### Explanation")
                            st.write(explanation_part)
                            
                        except Exception as e:
                            st.error(f"Error executing code: {str(e)}")
                    except Exception as e:
                        st.error(f"Error parsing response: {str(e)}")
            else:
                st.warning("Please enter a question!")
    
    with tab2:
        st.write("### Create visualizations")
        viz_request = st.text_input("Enter your visualization request (e.g., 'Create a scatter plot of X vs Y' or 'Show distribution of column Z')")
        
        if st.button("Generate Plot", key="viz_button"):
            if viz_request:
                with st.spinner("Generating visualization..."):
                    # Get columns as string
                    columns_str = ", ".join(st.session_state.data.columns.tolist())
                    
                    # Get response from LLM
                    response = viz_chain.run(columns=columns_str, request=viz_request)
                    
                    # Extract code
                    try:
                        code_part = extract_code_from_response(response)
                        
                        # Display code
                        st.code(code_part, language="python")
                        
                        # Execute code and display plot
                        try:
                            exec_locals = {
                                "pd": pd, 
                                "px": px, 
                                "go": go, 
                                "df": st.session_state.data,
                                "st": st
                            }
                            code_to_run = clean_code_for_streamlit(code_part)
                            exec(code_to_run, {}, exec_locals)
                        except Exception as e:
                            st.error(f"Error executing code: {str(e)}")
                    except Exception as e:
                        st.error(f"Error parsing response: {str(e)}")
            else:
                st.warning("Please enter a visualization request!")
else:
    st.info("Please upload a dataset to begin!") 