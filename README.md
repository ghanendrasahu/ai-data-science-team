# AI Data Science Team 🤖📊

An autonomous multi-agent AI system that performs end-to-end data science workflows using Large Language Models, Agentic AI, and Machine Learning.

The goal of this project is to build an AI-powered data science team that can analyze datasets, perform exploratory analysis, clean data, engineer features, train ML models, evaluate results, and generate business insights with minimal human intervention.

---

# 🚀 Project Overview

Traditional data science workflows require multiple roles:

* Data Analyst → explores data
* Data Engineer → cleans and prepares data
* ML Engineer → trains models
* Data Scientist → evaluates results
* Business Analyst → creates reports

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

# ✨ Features

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

## Machine Learning Agent

* Model selection
* Training pipeline creation
* Multiple model comparison
* Hyperparameter tuning

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

# 🧠 AI Architecture

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

# 🛠 Tech Stack

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

# 📁 Project Structure

```
ai-data-science-team/

│
├── app/
│   ├── agents/
│   │   ├── planner.py
│   │   ├── eda.py
│   │   ├── cleaning.py
│   │   ├── feature.py
│   │   ├── trainer.py
│   │   ├── evaluator.py
│   │   ├── report.py
│   │   └── chatbot.py
│   │
│   ├── graph/
│   │   ├── workflow.py
│   │   └── state.py
│   │
│   ├── tools/
│   │   ├── csv_tool.py
│   │   ├── pandas_tool.py
│   │   ├── plot_tool.py
│   │   └── sklearn_tool.py
│   │
│   ├── services/
│   │   └── llm.py
│   │
│   └── main.py
│
├── frontend/
│   └── app.py
│
├── uploads/
├── models/
├── reports/
│
├── requirements.txt
├── Dockerfile
├── .env
└── README.md
```

---

# ⚙️ Installation

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

# 🔑 Environment Setup

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

# ▶️ Running the Application

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

# 🗺 Development Roadmap

## Phase 1 — Foundation

* [x] Project structure
* [x] FastAPI setup
* [x] LLM connection
* [x] Configuration management

## Phase 2 — Data Processing

* [x] Dataset upload
* [x] Data profiling
* [x] Cleaning pipeline

## Phase 3 — Agent Development

* [x] Planner Agent
* [x] EDA Agent
* [x] Feature Agent
* [x] ML Agent
* [x] Evaluation Agent

## Phase 4 — Agent Orchestration

* [x] LangGraph workflow
* [x] State management
* [x] Memory

## Phase 5 — Production

* [x] Streamlit UI
* [x] Docker
* [x] Deployment
* [ ] Monitoring

---

# 🎯 Future Improvements

* Add MCP tools
* Add vector database memory
* Add SQL database agent
* Add automated feature selection
* Add model deployment agent
* Add GitHub integration
* Add automated experiment tracking with MLflow

---

# 👨‍💻 Author

Built as a production-style GenAI + Agentic AI portfolio project.

Technologies demonstrated:

* Generative AI
* Agentic AI
* RAG
* LLM Applications
* Machine Learning
* MLOps Concepts
* AI System Design

---
