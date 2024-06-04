'''
===========================================
        Module: Open-source LLM Setup
===========================================
'''
from langchain.llms import CTransformers
from dotenv import find_dotenv, load_dotenv
import box
import yaml

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open(r'D:\vs files\RAG-LLAMA2-Streamlit-FAISS-master\config\config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


def build_llm():
        
        print(f"MODEL_BIN_PATH: {cfg.MODEL_BIN_PATH}")
        print(f"MODEL_TYPE: {cfg.MODEL_TYPE}")
        print(f"MAX_NEW_TOKENS: {cfg.MAX_NEW_TOKENS}")
        print(f"TEMPERATURE: {cfg.TEMPERATURE}")
        # Local CTransformers model
        llm = CTransformers(
                model=cfg.MODEL_BIN_PATH,
                model_type=cfg.MODEL_TYPE,
                config={
                        'max_new_tokens': cfg.MAX_NEW_TOKENS,
                        'temperature': cfg.TEMPERATURE
                        } 
        )
        return llm
