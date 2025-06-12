import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os
import re
import traceback
import sklearn
from sklearn import decomposition, preprocessing, cluster, metrics, model_selection

# Utility Functions (Shared) 
def clean_code_for_streamlit(code: str) -> str:
    """
    Take a code string intended for a Jupyter or plain‐Python environment and
    convert recognized patterns (e.g., print()) into Streamlit equivalents 
    (st.write(), st.plotly_chart(), etc.). Strip out obvious non‐code lines,
    remove any stray backticks, and force df = st.session_state.data if re‐defined.
    """
    # Remove any stray backticks that might wrap inline code
    code = code.replace('`', '')
    code = code.replace('print(', 'st.write(')
    code = re.sub(r'xlabel\s*=', 'xaxis_title=', code)
    code = re.sub(r'ylabel\s*=', 'yaxis_title=', code)
    lines = code.splitlines()
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Remove markdown fences, comments that cannot run in Streamlit, or bare pd.read_csv
        if (
            stripped.startswith('#')
            or stripped.startswith('```')
            or 'pd.read_csv' in stripped
            or stripped == ''
        ):
            continue
        # If the code tries to reassign df, override it so it always uses st.session_state.data
        if stripped.startswith('df ='):
            cleaned_lines.append('df = st.session_state.data')
            continue
        cleaned_lines.append(line)
    return '\n'.join(cleaned_lines)

def extract_code_from_response(response: str) -> str:
    """
    Given a multi‐section LLM response, locate the first ```python``` fenced block 
    or a "Code:" prefix block, and return just the code inside.
    """
    match = re.search(r'```(?:python)?(.*?)```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match2 = re.search(r'Code:\s*(.+?)\s*(?:Explanation:|$)', response, re.IGNORECASE | re.DOTALL)
    if match2:
        return match2.group(1).strip()
    
    # gather any lines that look like code
    lines = response.splitlines()
    code_lines = []
    in_code_block = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block or line.strip().endswith(":") or line.strip().startswith(("df", "for ", "if ", "result")):
            code_lines.append(line)
    return "\n".join(code_lines).strip()

def extract_explanation_from_response(response: str) -> str:
    """
    After extracting the code portion, grab everything after "Explanation:" 
    or return an empty string if no explicit marker found.
    """
    match = re.search(r'Explanation:\s*(.*)', response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

if "data" not in st.session_state:
    st.session_state.data = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# Gemini (Flash 2.5 Preview) with Structured Output Prompts
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-preview-05-20",
    temperature=0.3,
    max_output_tokens=2048,
    max_retries=2,
)

# Prompt Templates 
eda_template = """You are a seasoned data analysis expert. 
I want you to show your reasoning step by step. Always prefix each thought with "Thought:" 
and each action with "Action:". Then wrap your code in a "Code: ..." block, followed by "Explanation: ...".

For example:
  Thought: <describe your next move>\n
  Action: <which tool you call>["<tool input>"]\n
  Observation: <what that tool returned>\n
  Thought: <next move>\n
  …\n  
Finally, you must output:
Code: <python code here> \n
Explanation: <natural-language reasoning here> \n

The code must use pandas (imported as pd) to answer the question, referencing the DataFrame named `df`. 
You may also use scikit-learn (imported as sklearn) and its major submodules (decomposition, preprocessing, cluster, metrics, model_selection) for advanced analytics (e.g., PCA, clustering). 

**Instructions for the generated code:**
1.  Perform the analysis as requested by the user.
2.  Use `st.write()`, `st.dataframe()`, or other Streamlit functions to display any tables, plots, or intermediate results in the app.
3.  **Crucially, you must create a final string variable named `result` that contains a concise, natural-language summary of the answer to the user's question.** This summary will be passed back to the main agent.
4.  **Note on dtypes**: If you need to display `df.dtypes`, convert the data types to strings first to avoid display errors. For example: `dtypes_df = df.dtypes.astype(str).reset_index(); dtypes_df.columns=['Column', 'Type']; st.dataframe(dtypes_df)`

Do not import pandas or re‐define df – assume df is already loaded. 
Dataset columns: {columns}
Question: {question}
"""

eda_prompt = PromptTemplate(
    input_variables=["columns", "question"],
    template=eda_template
)

viz_template = """You are a data visualization guru. 
I want you to show your reasoning step by step. Always prefix each thought with "Thought:" 
and each action with "Action:". Then wrap your code in a "Code: ..." block, followed by "Explanation: ...".

For example:
  Thought: <describe your next move>\n
  Action: <which tool you call>[""<tool input>"">]\n
  Observation: <what that tool returned>\n
  Thought: <next move>\n
  …\n  
When I ask for a visualization, finally, you must output:

Code: <code here> \n
Explanation: <brief explanation of the chart> \n
The code must use either `plotly.express as px` or `plotly.graph_objects as go` 
and must call `st.plotly_chart(fig)` at the end so the figure appears in Streamlit.
You may also use scikit-learn (imported as sklearn) and its major submodules (decomposition, preprocessing, cluster, metrics, model_selection) for advanced analytics (e.g., PCA, clustering).
Do not import plotly or re‐define df – assume df is already loaded.  
Dataset columns: {columns}
Request: {request}
"""

viz_prompt = PromptTemplate(
    input_variables=["columns", "request"],
    template=viz_template
)

#Chains
eda_chain = LLMChain(llm=llm, prompt=eda_prompt)
viz_chain = LLMChain(llm=llm, prompt=viz_prompt)

# EDA Tool (Pandas Code Generator + Executor)
def eda_tool_func(query: str) -> str:
    """
    1. Ask Gemini to generate pure pandas code plus explanation for the EDA query.
    2. Extract the code block, clean it for Streamlit, and detect if it's a bare expression
       (no assignment, no st.write). If so, wrap it in "result = <expr>" and then st.write(result).
       Otherwise, let the code run as provided, appending "st.write(<var>)" if there's an assignment
       to a known variable name.
    3. Execute that code in a namespace where df = st.session_state.data, so that st.write(...) 
       will actually display in Streamlit.
    4. Capture the computed value (if any) in `exec_namespace["result"]` and return a string of the form:
       "result: <value>\nExplanation: <explanation text>" so the agent sees both.
    """
    df = st.session_state.data.copy()
    columns_str = ", ".join(df.columns.tolist())

    # code + explanation from LLM
    llm_response = eda_chain.run(columns=columns_str, question=query)
    st.write("#### [EDA Tool] LLM Raw Response:")
    st.write(llm_response)

    # Extract
    code_part = extract_code_from_response(llm_response)
    explanation_part = extract_explanation_from_response(llm_response)
    if not explanation_part:
        explanation_part = "The code was executed to generate the output seen in the Streamlit app."

    # Prepare for execution
    exec_namespace = {"pd": pd, "df": df, "st": st, "sklearn": sklearn, "decomposition": decomposition, "preprocessing": preprocessing, "cluster": cluster, "metrics": metrics, "model_selection": model_selection}
    cleaned = clean_code_for_streamlit(code_part).strip()

    # If any bare expression (no assignment, no "st.write")
    is_bare_expr = ("=" not in cleaned) and ("st.write" not in cleaned) and bool(cleaned)
    if is_bare_expr:
        cleaned = f"result = {cleaned}\nst.write(result)"
    

    # Execute cleaned code
    try:
        exec(cleaned, {}, exec_namespace)
    except Exception as err:
        st.error(f"An error occurred while executing the generated code for the EDA tool.")
        st.code(cleaned, language="python")
        tb = traceback.format_exc()
        return f"Error executing EDA code:\n```\n{tb}\n```"

    # Capture result if it exists
    result_val = exec_namespace.get("result", None)
    if result_val is not None:
        # Return actual value plus explanation
        return f"result: {result_val}\nExplanation: {explanation_part}"
    else:
        # If no result variable was explicitly created, return a generic message.
        return f"result: (see Streamlit output for details)\nExplanation: {explanation_part}"

eda_tool = Tool(
    name="EDA Tool",
    func=eda_tool_func,
    description="Use this to answer questions or perform analysis on the dataset. It is the primary tool for any task that requires generating pandas or scikit-learn code. It performs the analysis, displays detailed results in the app, and **returns a concise text summary** of the findings. Ideal for tasks like 'what is the correlation?' or 'perform PCA and summarize the results'."
)

# Visualization Tool (Plotly Code Generator + Executor)
def viz_tool_func(request: str) -> str:
    """
    1. Ask Gemini to generate Plotly code (with `fig = ...` and then `st.plotly_chart(fig)`).
    2. Extract and clean that code, run it, and thus embed the chart live in Streamlit.
    3. Return "result: (see Streamlit chart)\nExplanation: <...>" so the agent knows
       that the chart appeared in the UI and can include the explanation.
    """
    df = st.session_state.data.copy()
    columns_str = ", ".join(df.columns.tolist())
    llm_response = viz_chain.run(columns=columns_str, request=request)
    st.write("#### [Viz Tool] LLM Raw Response:")
    st.write(llm_response)

    code_part = extract_code_from_response(llm_response)
    explanation_part = extract_explanation_from_response(llm_response)

    exec_namespace = {"pd": pd, "px": px, "go": go, "df": df, "st": st, "sklearn": sklearn, "decomposition": decomposition, "preprocessing": preprocessing, "cluster": cluster, "metrics": metrics, "model_selection": model_selection}
    cleaned = clean_code_for_streamlit(code_part).strip()

    # If user forgot st.plotly_chart but created `fig`, append it
    if "st.plotly_chart" not in cleaned and "fig" in cleaned:
        cleaned += "\nst.plotly_chart(fig)"

    try:
        exec(cleaned, {}, exec_namespace)
    except Exception as err:
        st.error(f"An error occurred while executing the generated code for the Visualization tool.")
        st.code(cleaned, language="python")
        tb = traceback.format_exc()
        return f"Error executing visualization code:\n```\n{tb}\n```"

    return f"result: (see Streamlit chart)\nExplanation: {explanation_part}"

viz_tool = Tool(
    name="Visualization Tool",
    func=viz_tool_func,
    description="Use this tool to produce any Plotly visualization on the dataset. scikit-learn (sklearn) and its major submodules (decomposition, preprocessing, cluster, metrics, model_selection) are also available for advanced analytics (e.g., PCA, clustering). Returns a reference to the chart and explanation text."
)

# Python Execution Tool (Raw Interpreter)
def python_exec_tool_func(code: str) -> str:
    """
    A "raw Python REPL" tool. Given arbitrary Python code, execute it in a namespace
    where df, pd, px, go, and st already exist. Return either the printed result 
    or an error message. If the code creates a variable called `result`, return that value explicitly.
    """
    df = st.session_state.data.copy()
    namespace = {"pd": pd, "px": px, "go": go, "df": df, "st": st, "sklearn": sklearn, "decomposition": decomposition, "preprocessing": preprocessing, "cluster": cluster, "metrics": metrics, "model_selection": model_selection}
    cleaned = clean_code_for_streamlit(code).strip()

    # Detect bare expression: wrap in st.write(...) and capture as result
    is_bare_expr = ("=" not in cleaned) and ("st.write" not in cleaned) and bool(cleaned)
    if is_bare_expr:
        cleaned = f"result = {cleaned}\nst.write(result)"
    else:
        # If assignment but no st.write, append st.write on assigned var
        if ("=" in cleaned) and ("st.write" not in cleaned):
            first_line = cleaned.splitlines()[0]
            lhs = first_line.split("=", 1)[0].strip()
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", lhs):
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


python_exec_tool = Tool(
    name="Python Execution Tool",
    func=python_exec_tool_func,
    description="A simple Python REPL. Use it for quick, single-line calculations or manipulations (e.g., 'df.shape', 'df.columns'). It executes code and prints to the app, but **it does not return a summary of the analysis**. For any multi-step analysis or questions that need an explanation, use the 'EDA Tool' instead."
)

# Assemble All Tools into a Single LangChain Agent
tools = [eda_tool, viz_tool, python_exec_tool]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,                  # Show chain‐of‐thought reasoning on screen
    handle_parsing_errors=True     # Retry if the LLM's output fails to parse
)

# Streamlit UI
st.title("EDA Agent (Fully‐Orchestrated with Chains + Python Interpreter)")

st.write(
    """
    **Instructions:**  
    1. Upload a CSV file.  
    2. Once it's loaded, you can ask *any* question or request a visualization.  
    3. The agent has three tools under the hood:  
       - **EDA Tool (Pandas + scikit-learn)** for pure pandas analysis and advanced analytics (PCA, clustering, etc.).  
       - **Visualization Tool (Plotly + scikit-learn)** for creating charts and advanced visualizations.  
       - **Python Execution Tool** to run arbitrary code.  
    4. The agent's chain‐of‐thought will appear live below as it thinks.  
    5. The output (numbers, tables, or charts) will be rendered in the app.
    """
)

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None and (st.session_state.file_name != uploaded_file.name):
    st.session_state.data = pd.read_csv(uploaded_file)
    st.session_state.file_name = uploaded_file.name
    st.success(f"📥 File '{uploaded_file.name}' uploaded successfully!")

if st.session_state.data is not None:
    st.write("### Dataset Preview:")
    st.dataframe(st.session_state.data.head())

    user_input = st.text_input(
        'Ask a question or request a visualization\n(e.g., "What is the mean of column A?" or "Plot column X vs Y")'
    )
    if st.button("Run Agent"):
        with st.spinner("Agent is thinking..."):
            callback_handler = StreamlitCallbackHandler(st.container())
            try:
                final_answer = agent.run(user_input, callbacks=[callback_handler])
            except Exception as e:
                final_answer = f"❗ Agent encountered an unrecoverable error: {e}"
        st.write("### Agent's Final Answer:")
        st.write(final_answer)
else:
    st.info("Upload a dataset to begin")
