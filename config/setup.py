import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def setup_environment():
    """
    Sets up the environment variables and returns the configured LLM.
    """
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY", "")

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-preview-05-20",
        temperature=0.3,
        max_output_tokens=2048,
        max_retries=2,
    )
    
    return llm 