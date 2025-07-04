import os
import tiktoken
from dotenv import load_dotenv

load_dotenv()

# --- LLM and API Configuration ---
try:
    ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
except KeyError:
    raise ValueError("ANTHROPIC_API_KEY not found in .env file. Please add it.")

LLM_MODEL = "claude-3-haiku-20240307" # "claude-sonnet-4-0"

# --- Tokenizer Configuration ---
ENCODING_NAME = "cl100k_base"
ENCODING = tiktoken.get_encoding(ENCODING_NAME)
MAX_TOKENS = 2048  # Chunk max tokens
TARGET_TOKENS = 1500  # Target for chunk size before recursion

# --- Chunking Configuration ---
MONOLOGUE_OVERLAP_SENTENCES = 2  # Number of sentences for overlap in monologues

# --- File and Directory Paths ---
LOGS_DIR = "logs"
LOG_FILE_PATH = os.path.join(LOGS_DIR, "chunking_debug.log")
INPUT_JSON_PATH = 'data/test-transcript-data.json'
OUTPUT_JSON_PATH = "data/semantic_chunks.json"

# --- Logging Configuration ---
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s" 