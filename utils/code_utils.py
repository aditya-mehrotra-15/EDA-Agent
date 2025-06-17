# utils.py
import re

def extract_code(response: str) -> str:
    """
    Pull out the first Python code block (```python ... ```) or a
    `Code:` section from an LLM response.
    """
    # 1) Look for a ```python fenced block (with or without colon)
    m = re.search(r'```python:?(.*?)```', response, re.DOTALL)
    if not m:
        # 2) Fallback to any ```...``` block
        m = re.search(r'```(.*?)```', response, re.DOTALL)
    if m:
        code = m.group(1).strip()
    else:
        # 3) Finally, look for a leading "Code:" until "Explanation:"
        m = re.search(r'Code:\s*(.*?)\s*(?:Explanation:|$)', response, re.DOTALL)
        code = m.group(1).strip() if m else ""
    
    # Clean up the code: remove any embedded "Code:" markers
    if code:
        # Remove any "Code:" that might be embedded in the middle of the code
        code = re.sub(r'\n\s*Code:\s*', '\n', code)
        code = re.sub(r'^\s*Code:\s*', '', code)
        # Remove any trailing "Explanation:" markers
        code = re.sub(r'\n\s*Explanation:\s*.*$', '', code, flags=re.DOTALL)
    
    return code


def extract_explanation(response: str) -> str:
    """
    Pull out the text after an 'Explanation:' marker, if present.
    """
    m = re.search(r'Explanation:\s*(.*)', response, re.DOTALL)
    return m.group(1).strip() if m else ""
