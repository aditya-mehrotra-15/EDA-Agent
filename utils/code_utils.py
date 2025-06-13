import re

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