# AI Data Science Team ðŸ¤–ðŸ“Š

An autonomous multi-agent AI system that performs end-to-end data science workflows using Large Language Models, Agentic AI, and Machine Learning.

The goal of this project is to build an AI-powered data science team that can analyze datasets, perform exploratory analysis, clean data, engineer features, train ML models, evaluate results, and generate business insights with minimal human intervention.

---

# ðŸš€ Project Overview

Traditional data science workflows require multiple roles:

* Data Analyst â†’ explores data
* Data Engineer â†’ cleans and prepares data
* ML Engineer â†’ trains models
* Data Scientist â†’ evaluates results
* Business Analyst â†’ creates reports

This project combines these responsibilities into a team of AI agents.

## Workflow

```
                 User
                   |
                   |
            Upload Dataset
                   |
                   v
            Planner Agent
                   |
    --------------------------------
    |              |               |
    v              v               v
EDA Agent    Cleaning Agent   Feature Agent
    |              |               |
    --------------------------------
                   |
                   v
          ML Training Agent
                   |
                   v
          Evaluation Agent
                   |
                   v
        Report Generation Agent
                   |
                   v
          Dataset Chat Agent
```

---

# âœ¨ Features

## Dataset Understanding

* Upload CSV datasets
* Automatic dataset profiling
* Column type detection
* Missing value analysis
* Data quality checks

## Exploratory Data Analysis Agent

* Statistical summaries
* Distribution analysis
* Correlation detection
* Automatic visualization generation

## Data Cleaning Agent

* Missing value handling
* Duplicate detection
* Outlier identification
* Data transformation

## Feature Engineering Agent

* Feature suggestions
* Encoding categorical variables
* Scaling numerical features
* Feature importance analysis
* **Automated feature selection** (keeps the strongest `MAX_FEATURES` features)

## Machine Learning Agent

* Model selection
* Training pipeline creation
* Multiple model comparison
* Hyperparameter tuning
* **Experiment tracking** (SQLite built-in + optional MLflow)

Supported models:

* Logistic Regression
* Random Forest
* XGBoost
* Decision Trees
* KNN
* SVM

## Evaluation Agent

* Performance metrics
* Model comparison
* Error analysis
* Model recommendation

## SQL Agent

* Natural-language questions â†’ read-only SELECT queries
* Aggregate / filter / group analysis on uploaded datasets
* Safe in-memory SQLite execution (only SELECT allowed)

## Model Deployment Agent

* Packages the winning model into a self-contained bundle
* `schema.json`, `predict.py` and a copy of the model artifact
* `POST /deploy/{run_id}` + `POST /predict` endpoints
* Optional GitHub report export (`POST /export/github/{run_id}`)

## Vector Memory

* Stores run summaries as embeddings (TF-IDF)
* Semantic retrieval across previous runs via `POST /memory/search`
* Lets agents remember what has been learned before (cross-run memory)

## Monitoring

* `GET /monitoring/overview` (run counts, statuses, memory size)
* `GET /runs/recent` (per-run summaries + metrics)
* `POST /memory/search`

## AI Report Generator

Creates:

* Technical report
* Business insights
* Model explanation
* Recommendations

## Dataset Chat Assistant

Users can ask:

```
"What are the important features?"

"Why did this model perform better?"

"Explain the dataset in simple terms."
```

---

# ðŸ§  AI Architecture

## Agent Framework

* LangGraph
* LangChain

## Agents

| Agent            | Responsibility            |
| ---------------- | ------------------------- |
| Planner Agent    | Creates workflow plan     |
| EDA Agent        | Analyzes dataset          |
| Cleaning Agent   | Prepares data             |
| Feature Agent    | Creates features          |
| ML Agent         | Trains models             |
| Evaluation Agent | Selects best model        |
| Report Agent     | Generates insights        |
| Chat Agent       | Answers dataset questions |

---

# ðŸ›  Tech Stack

## Backend

* Python
* FastAPI
* Pydantic

## AI

* LangGraph
* LangChain
* OpenAI / Anthropic / Ollama

## Machine Learning

* Pandas
* NumPy
* Scikit-learn
* XGBoost

## Visualization

* Plotly
* Matplotlib

## Frontend

* Streamlit

## Database

* SQLite

## Deployment

* Docker
* Cloud Deployment

---

# ðŸ“ Project Structure

```
ai-data-science-team/

â”‚
â”œâ”€â”€ app/
â”‚   â”œâ”€â”€ agents/
â”‚   â”‚   â”œâ”€â”€ planner.py
â”‚   â”‚   â”œâ”€â”€ eda.py
â”‚   â”‚   â”œâ”€â”€ cleaning.py
â”‚   â”‚   â”œâ”€â”€ feature.py
â”‚   â”‚   â”œâ”€â”€ trainer.py
â”‚   â”‚   â”œâ”€â”€ evaluator.py
â”‚   â”‚   â”œâ”€â”€ report.py
â”‚   â”‚   â””â”€â”€ chatbot.py
â”‚   â”‚
â”‚   â”œâ”€â”€ graph/
â”‚   â”‚   â”œâ”€â”€ workflow.py
â”‚   â”‚   â””â”€â”€ state.py
â”‚   â”‚
â”‚   â”œâ”€â”€ tools/
â”‚   â”‚   â”œâ”€â”€ csv_tool.py
â”‚   â”‚   â”œâ”€â”€ pandas_tool.py
â”‚   â”‚   â”œâ”€â”€ plot_tool.py
â”‚   â”‚   â””â”€â”€ sklearn_tool.py
â”‚   â”‚
â”‚   â”œâ”€â”€ services/
â”‚   â”‚   â”œâ”€â”€ llm.py
â”‚   â”‚   â”œâ”€â”€ store.py
â”‚   â”‚   â”œâ”€â”€ tracking.py
â”‚   â”‚   â”œâ”€â”€ memory.py
â”‚   â”‚   â””â”€â”€ github.py
â”‚   â”‚
â”‚   â”œâ”€â”€ config.py
â”‚   â””â”€â”€ main.py
â”‚
â”œâ”€â”€ frontend/
â”‚   â””â”€â”€ app.py

â”œâ”€â”€ uploads/
â”œâ”€â”€ models/
â”œâ”€â”€ reports/
â”œâ”€â”€ deployments/
â”‚
â”œâ”€â”€ requirements.txt
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ .env.example
â””â”€â”€ README.md
```

---

# âš™ï¸ Installation

## Clone Repository

```bash
git clone <repository-url>

cd ai-data-science-team
```

---

## Create Virtual Environment

```bash
python -m venv .venv
```

Activate:

Windows:

```bash
.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ðŸ”‘ Environment Setup

Create `.env`

Example:

```
OPENAI_API_KEY=your_api_key
```

or for local models:

```
OLLAMA_MODEL=llama3
```

---

# â–¶ï¸ Running the Application

## Start Backend

```bash
uvicorn app.main:app --reload
```

API:

```
http://localhost:8000
```

---

## Start Frontend

```bash
streamlit run frontend/app.py
```

---

# ðŸ—º Development Roadmap

## Phase 1 â€” Foundation

* [x] Project structure
* [x] FastAPI setup
* [x] LLM connection
* [x] Configuration management

## Phase 2 â€” Data Processing

* [x] Dataset upload
* [x] Data profiling
* [x] Cleaning pipeline

## Phase 3 â€” Agent Development

* [x] Planner Agent
* [x] EDA Agent
* [x] Feature Agent
* [x] ML Agent
* [x] Evaluation Agent

## Phase 4 â€” Agent Orchestration

* [x] LangGraph workflow
* [x] State management
* [x] Memory

## Phase 5 â€” Production

* [x] Streamlit UI
* [x] Docker
* [x] Deployment
* [x] Monitoring

## Phase 6 â€” Platform Extensions

* [x] MCP tools (stdio server via `python -m app.mcp_server`)
* [x] Vector database memory (cross-run retrieval)
* [x] SQL database agent
* [x] Automated feature selection
* [x] Model deployment agent
* [x] GitHub report export (optional)
* [x] Experiment tracking (optional MLflow)

---

# ðŸŽ¯ Future Improvements

* Add vector database memory with real embeddings (FAISS / Chroma)
* Add automated feature selection with tree-based importance
* Add automated experiment tracking with MLflow

---

# ðŸ‘¨â€ðŸ’» Author

Built as a production-style GenAI + Agentic AI portfolio project.

Technologies demonstrated:

* Generative AI
* Agentic AI
* LLM Applications
* Machine Learning
* MLOps Concepts
* AI System Design

---

## License

MIT (c) 2026 ghanendrasahu - see [LICENSE](LICENSE).

Issues and pull requests are welcome.
