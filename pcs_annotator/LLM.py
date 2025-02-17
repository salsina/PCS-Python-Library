import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
# from langchain_huggingface import HuggingFaceEndpoint
from huggingface_hub import InferenceClient  # Use InferenceClient instead of HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEndpoint

# from langchain_huggingface import HuggingFacePipeline


class LLM:
    def __init__(self, model_name):
        load_dotenv()
        self.model_name = model_name
        self.llm = self._initialize_model()
        parser = StrOutputParser()
        self.llm = self.llm | parser

    def _initialize_model(self):
        """
        Initialize the appropriate LLM model based on the model_name.
        """
        if self.model_name.startswith("llama"):  # e.g., llama-3.1-8b-instant
            key = os.getenv('GROQ_API_KEY')
            if not key:
                raise ValueError("Missing GROQ_API_KEY in environment variables.")
            model = ChatGroq(api_key=key, model_name=self.model_name, temperature=0.1)

        elif self.model_name.startswith("gpt"):  # e.g., gpt-4
            key = os.getenv('OPENAI_API_KEY')
            if not key:
                raise ValueError("Missing OPENAI_API_KEY in environment variables.")
            model = ChatOpenAI(api_key=key, model=self.model_name, temperature=0.1)

        elif self.model_name.startswith("claude"):  # e.g., claude-3-5-sonnet-20241022
            key = os.getenv('ANTHROPIC_API_KEY')
            if not key:
                raise ValueError("Missing ANTHROPIC_API_KEY in environment variables.")
            model = ChatAnthropic(api_key=key, model=self.model_name, temperature=0.1)
        
        elif self.model_name.startswith("mistral"):  # e.g., mistralai/Mistral-7B-Instruct-v0.3
            key = os.getenv('HUGGINGFACE_API_KEY')
            
            if not key:
                raise ValueError("Missing HUGGINGFACE_API_KEY in environment variables.")

            model = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.3",
                task="text-generation",
                temperature=0.1,
                huggingfacehub_api_token=key,
            )
            
            

        elif self.model_name.startswith("google"):  # e.g., gemma-2-9b-it
            key = os.getenv('HUGGINGFACE_API_KEY')
            if not key:
                raise ValueError("Missing HUGGINGFACE_API_KEY in environment variables.")

            model = HuggingFaceEndpoint(
                repo_id="google/gemma-2-9b-it",
                task="text-generation",
                temperature=0.1,
                huggingfacehub_api_token=key,
            )

        else:
            raise ValueError(f"Unsupported model name: {self.model_name}")

        return model
