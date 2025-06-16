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
        return m.group(1).strip()

    # 3) Finally, look for a leading "Code:" until "Explanation:"
    m = re.search(r'Code:\s*(.*?)\s*(?:Explanation:|$)', response, re.DOTALL)
    return m.group(1).strip() if m else ""


def extract_explanation(response: str) -> str:
    """
    Pull out the text after an 'Explanation:' marker, if present.
    """
    m = re.search(r'Explanation:\s*(.*)', response, re.DOTALL)
    return m.group(1).strip() if m else ""
