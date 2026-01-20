# 🧠 AI Engineering Bootcamp: RAG & Evaluation System

> **A "First Principles" implementation of Retrieval-Augmented Generation (RAG) designed to demonstrate core engineering skills, algorithmic understanding, and CI/CD automation.**

## 🎯 Project Goals (Why this exists)
This repository is not just another LangChain wrapper. It was built to satisfy specific JD requirements for an **AI Engineer** role, focusing on:
1.  **Solid Engineering Foundation**: Structured as a production-grade Python package (`src` layout, `pyproject.toml`).
2.  **Algorithmic Depth**: Implements **Vector Search** and **Tokenization** from scratch (Zero Dependencies) to demonstrate understanding of RAG internals.
3.  **CI/CD & Quality**: Includes a complete GitHub Actions pipeline for automated testing, linting (Ruff), and type checking (Mypy).
4.  **Evaluation**: Implements a metric for **"Hallucination Detection"** (Faithfulness Score).

---

## 🏗️ Architecture

### 1. The Core (Pure Python RAG)
Located in `src/rag_system/core.py`.
- **Dependency-Free**: Bypasses typical library compatibility issues (e.g., Python 3.14 preview conflicts).
- **Custom Vector Store**: Implements an in-memory document store with hash-based embeddings.
- **Math-Based Retrieval**: Manually calculates Cosine Similarity.

### 2. The Quality Gate
- **Linter**: `Ruff` (configured for strict Modern Python standards).
- **Type Checker**: `Mypy` (Static type analysis).
- **Testing**: `unittest` suite (Zero dependency testing).

### 3. CI Pipeline
Defined in `.github/workflows/ci.yml`. automatically runs the Quality Gate on every push.

---

## 🚀 How to Run

### Installation
This project uses a modern `pyproject.toml` setup.

```bash
# 1. Create a virtual environment (Recommended)
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Mac/Linux
source .venv/bin/activate

# 2. Install in editable mode
pip install -e .
```

### Run the Demo
The demo digests a knowledge base about an AI agent and answers questions.

```bash
python src/main.py
```

### Run Tests
```bash
python -m unittest discover tests
```

---

## 📊 Evaluation Metric (Hallucination Detection)

The system includes a **Faithfulness Score** (`eval_score`).
It calculates the overlap between significant tokens in the **Generated Answer** vs. the **Retrieved Context**.

```python
# Logic simplified:
Score = (Keywords in Answer found in Context) / (Total Keywords in Answer)
```

- **1.0**: High confidence (Answer is grounded in text).
- **0.0**: Potential Hallucination (Answer contains information not found in context).

---

## 📂 Project Structure

```text
.
├── .github/workflows/  # CI/CD Pipeline
├── src/
│   └── rag_system/     # Core Logic
│       ├── core.py     # RAG Implementation (The "Brain")
│       └── __init__.py
├── tests/              # Automated Tests
├── pyproject.toml      # Project Configuration (Deps, Tools)
└── README.md           # You are here
```
