# File: config.py
# Single source of truth for all paths, constants, model names, and
# clause-family definitions. Every other module imports from here.
# Nothing is hardcoded in any other file.

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file so ANTHROPIC_API_KEY is available via os.environ
# anywhere in the project that imports config.
load_dotenv()

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"

# ---------------------------------------------------------------------------
# Raw CUAD input (gitignored; user provides via Git LFS)
# full_contracts_txt/ and master_clauses.csv must be present before
# running scripts/prepare_data.py.
# ---------------------------------------------------------------------------

CUAD_RAW_DIR = DATA_DIR / "cuad_raw"
FULL_CONTRACTS_DIR = CUAD_RAW_DIR / "full_contracts_txt"
MASTER_CLAUSES_CSV = CUAD_RAW_DIR / "master_clauses.csv"

# ---------------------------------------------------------------------------
# Split outputs (written by scripts/prepare_data.py)
# ---------------------------------------------------------------------------

REFERENCE_DIR = DATA_DIR / "reference"         # 80 reference .txt files
TEST_DIR = DATA_DIR / "test"                   # 20 test .txt files
REFERENCE_ANNOTATIONS_PATH = DATA_DIR / "reference_annotations.json"

# HELD OUT — only scripts/evaluate.py reads this.
# No other module may import or reference this path.
TEST_ANNOTATIONS_PATH = DATA_DIR / "test_annotations.json"

# ---------------------------------------------------------------------------
# Persisted vector store (gitignored; written by scripts/build_index.py)
# ---------------------------------------------------------------------------

CHROMA_DB_PATH = str(DATA_DIR / "chroma_db")

# ---------------------------------------------------------------------------
# Report outputs (written by agent/output_writer.py)
# ---------------------------------------------------------------------------

OUTPUT_JSON_DIR = ROOT_DIR / "output" / "json"
OUTPUT_HTML_DIR = ROOT_DIR / "output" / "html"
TEMPLATES_DIR = ROOT_DIR / "templates"

# ---------------------------------------------------------------------------
# 80/20 split parameters
# Do not change SPLIT_SEED after build_index.py has been run.
# Changing the seed invalidates the ChromaDB index and requires a full rebuild.
# ---------------------------------------------------------------------------

TOTAL_CONTRACTS_TO_USE = 100    # 80 reference + 20 test
REFERENCE_COUNT = 80
TEST_COUNT = 20
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Embedding model
# Used by: agent/embedder.py (loaded once, shared across all callers)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ---------------------------------------------------------------------------
# Chunking
# Word-level counts used as a token-count proxy to avoid a tokenizer
# dependency at index time. ~256 words is a reasonable proxy for 256 tokens
# with all-MiniLM-L6-v2's 512-token max input length.
# Used by: scripts/build_index.py, agent/clause_discovery.py
# ---------------------------------------------------------------------------

CHUNK_SIZE_WORDS = 256
CHUNK_OVERLAP_WORDS = 32

# ---------------------------------------------------------------------------
# Clause discovery — Embedding Use 1
# Operates on the uploaded contract only. No ChromaDB access.
# Used by: agent/clause_discovery.py
# ---------------------------------------------------------------------------

# Number of top-scoring chunks retained before span merging.
DISCOVERY_TOP_K = 5

# Minimum averaged cosine score across anchor queries for clause_found=True.
# Below this threshold the clause is considered absent from the contract.
# 0.30 is deliberately conservative — raises recall at the cost of some precision.
# If evaluation shows low recall, lower this threshold first before touching anchors.
DISCOVERY_MIN_SCORE = 0.30

# ---------------------------------------------------------------------------
# Precedent retrieval — Embedding Use 2
# Operates on ChromaDB only. No access to raw contract text.
# Used by: agent/precedent_retrieval.py
# ---------------------------------------------------------------------------

# Number of similar precedents returned per clause family card.
PRECEDENT_SIMILAR_TOP_K = 3

# Number of candidate results over-fetched from ChromaDB before Claude
# selects the most risk-posture-contrasting ones.
PRECEDENT_CONTRAST_FETCH_K = 10

# Number of contrasting precedents returned after Claude's selection.
PRECEDENT_CONTRAST_RETURN_K = 3

# Maximum characters of a reference clause text passed to Claude when
# building the contrasting candidates prompt. Prevents context overflow
# for very long reference clauses.
CONTRAST_CLAUSE_TRUNCATION_CHARS = 800

# ---------------------------------------------------------------------------
# Structured interpretation — agent/interpretation.py
# Used to compute the heuristic confidence score for each clause card.
# ---------------------------------------------------------------------------

# Clause texts shorter than this word count receive a confidence penalty.
# A short extracted text likely means clause_discovery found a partial span.
INTERPRETATION_MIN_WORDS = 50

# Confidence is reduced by this amount for each field that could not be
# resolved by regex and required a Claude call.
INTERPRETATION_CONFIDENCE_PENALTY_PER_LLM_FIELD = 0.1

# Additional penalty applied when the clause text is shorter than
# INTERPRETATION_MIN_WORDS.
INTERPRETATION_SHORT_TEXT_PENALTY = 0.2

# ---------------------------------------------------------------------------
# LLM
# Used by: agent/interpretation.py, agent/precedent_retrieval.py,
#          agent/risk_rating.py, agent/summarizer.py
# ---------------------------------------------------------------------------

# claude-sonnet-4-6 used over Haiku because risk rationale requires
# multi-step reasoning across clause text, structured interpretation,
# and precedent comparisons simultaneously.
MODEL = "claude-sonnet-4-6"

# Applies to all Claude calls in the project. Risk rating and summarizer
# responses are structured JSON so 1024 tokens is sufficient.
# Interpretation calls are shorter — this is a safe ceiling for all uses.
MAX_TOKENS = 1024

# ---------------------------------------------------------------------------
# Output rendering — agent/output_writer.py
# ---------------------------------------------------------------------------

# Hex color codes for risk level badges in the HTML report.
RISK_COLOR_MAP = {
    "high":   "#c0392b",   # red
    "medium": "#e67e22",   # amber
    "low":    "#27ae60",   # green
    None:     "#7f8c8d",   # grey — used when risk rating generation failed
}

# ---------------------------------------------------------------------------
# Clause families and CUAD column mapping
# ---------------------------------------------------------------------------

CLAUSE_FAMILIES = ["assignment", "change_of_control", "termination", "exclusivity"]

# Maps internal family keys to master_clauses.csv column header(s).
# For "exclusivity", either column being non-empty means the clause exists.
# Texts from both columns are concatenated when building the reference annotation.
#
# IMPORTANT (R2): verify column names against the actual CSV before running
# build_index.py. Run this check once:
#   import pandas as pd
#   from config import MASTER_CLAUSES_CSV
#   print(pd.read_csv(MASTER_CLAUSES_CSV).columns.tolist())
# Then compare the output against the values below.
CUAD_COLUMN_MAP = {
    "assignment":        ["Anti-Assignment"],
    "change_of_control": ["Change Of Control"],
    "termination":       ["Termination For Convenience"],
    "exclusivity":       ["Exclusivity", "Non-Compete"],
}

# ChromaDB collection names — one per clause family.
# Must match between scripts/build_index.py and agent/precedent_retrieval.py.
# Defined here so both files import from config rather than redefining locally.
COLLECTION_NAMES = {
    "assignment":        "cuad_assignment",
    "change_of_control": "cuad_change_of_control",
    "termination":       "cuad_termination",
    "exclusivity":       "cuad_exclusivity",
}