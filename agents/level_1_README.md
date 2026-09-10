# Single-Tool Travel Info Agent

A lightweight, local Retrieval-Augmented Generation (RAG) travel agent built using **LangGraph**, **LangChain**, and **Ollama**. The application automatically scrapes Wikivoyage articles, processes and embeds the destination details into a local **Chroma DB** vector store, and exposes a tool-calling graph agent capable of answering travel queries via an interactive CLI interface.

---

## Architecture Overview

```
                  +-----------------------+
                  |  Wikivoyage Web Page  |
                  +-----------+-----------+
                              |
                     (AsyncHtmlLoader)
                              v
                  +-----------------------+
                  |  Text Chunking & DB   |
                  |    (Chroma Vector)    |
                  +-----------+-----------+
                              |
                              v
+-----------------------------------------------------------+
|                     LangGraph Agent                       |
|                                                           |
|    +------------+    tool call    +------------------+    |
|    |  llm_node  | --------------> | ToolExecutionNode|    |
|    +------------+                 +--------+---------+    |
|          ^                                 |              |
|          +---------------------------------+              |
|                     tool output                           |
+-----------------------------------------------------------+
```

---

## Features

* **Async Data Ingestion**: Uses `AsyncHtmlLoader` to fetch destination content from Wikivoyage.
* **Local Embeddings & Vector Storage**: Persists embedded content in Chroma DB using Ollama.
* **LangGraph Orchestration**: Controls state flow between an LLM decision node and custom tool execution nodes.
* **Single-Tool Calling**: Uses a `search_travel_info` LangChain tool to query local vector indices dynamically during model inference.
* **Interactive CLI**: Enables user interaction directly via command line.

---

## Prerequisites

* **Python**: 3.10+
* **Ollama**: Installed and running locally.

Ensure your preferred LLM and embedding models are pulled into Ollama:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

---

## Installation

1. **Clone the repository** (or save the script locally):
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```
2. **Install dependencies**:

   ```bash
      uv sync
   ```

3. **Activate Virtual Env**:

   ```bash
      uv venv
   ```
---

## Configuration

Create a `.env` file in the root directory to define your environment settings:

```env
LLM_MODEL=llama3.2
EMBEDDING_MODEL=nomic-embed-text
WIKI_CRAWLER_USER_AGENT=TravelInfoAgent/1.0 (contact@example.com)
```

---

## Project Structure

```text
.
├── agents/level_1.py                     # Primary script containing graph logic & CLI loop
├── .env                        # Environment variable specifications
└── data/                       # Chroma DB persistence directory (auto-created)
    └── level_1/
```

---

## Usage

Run the agent script directly:

```bash
python3 level_1.py
```

* On initial execution, the script automatically fetches Wikivoyage documents for configured locations (default: `Cornwall`), embeds them, and persists the vector database.
* Once initialized, type your query into the prompt loop.

### Example CLI Session

```text
Waiting DB..
DB Done..
mini Travel Assistant (type 'exit' to quit)
You: What are the main attractions in Cornwall?
Assistant: Based on the travel info, major highlights in Cornwall include coastal trails, historic fishing villages, beaches...

You: exit
```

---

## Customization

To index additional destinations, open the script and update the `UK_DESTINATIONS` list with any valid Wikivoyage slug:

```python
UK_DESTINATIONS = [
    "Cornwall",
    "London",
    "Edinburgh",
]
```