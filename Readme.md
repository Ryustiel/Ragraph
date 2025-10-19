# RAGraph : Relational Accessor Graph

A dynamic, multi-layered knowledge graph backend for advanced context retrieval, built on SQLAlchemy and pgvector. RAGraph models information as "Contents" linked via weighted "Accessors" (semantic entry points), enabling contextual, multi-step queries that go beyond simple vector search.

This project is structured as a python package which can be used as a component of actual RAG (**Retrieval Augmented Generation**) applications. It was originally designed to handle messy knowledge extracted by LLMs from documents and conversations.

---

## TLDR

-   **Structured Knowledge Graph:** Store information in a relational database where "Content" nodes (data chunks) are linked to various "Accessor" nodes (topics, tasks, concepts). Accessor nodes can be searched for using semantic queries.
-   **Multi-Step Retrieval:** Find information by first performing a vector search for relevant `Accessors` and then traversing their weighted `Edges` to retrieve associated `Content`. Content relevance is determined by the weight and number of edges that led to that Content from the relevant Accessors.
-   **Dynamic & Layered:** The graph architecture is defined in a single configuration file. Add new layers (`documentation`, `clients`, `projects`, ...) and the database schema adapts automatically at runtime.
-   **Clean Transactional API:** Use a custom (and experimental) Pydantic-based "Transaction" objects (`AccessorTransaction`, `ContentTransaction`) that behave similarly to sqlalchemy's  orm objects for database interaction and API compatibility.
-   **Powered by pgvector:** Leverages PostgreSQL and the `pgvector` extension for efficient and scalable vector similarity search.
-   **Designed for Complex Context:** Ideal for applications where a single piece of content can be relevant in multiple contexts and accessed through different semantic lenses. This might also work well with indexes built by LLMs.

---

## Tech Stack and Techniques

-   **Database & ORM**
    -   **SQLAlchemy 2.0:** Core of the data layer, used for ORM, session management, and schema definition.
    -   **PostgreSQL:** The underlying relational database.
    -   **pgvector:** PostgreSQL extension for storing and searching vector embeddings with high performance.
-   **Data Modeling & Validation**
    -   **Pydantic:** Used to define "Transaction" models that provide a clean, validated API for creating and manipulating graph entities.
-   **Vector Embeddings**
    -   **OpenAI SDK:** Integrates with Azure OpenAI services to generate text embeddings (e.g., `text-embedding-3-large`).
-   **Architecture & Patterns**
    -   **Dynamic ORM Generation:** SQLAlchemy models for the multiple `Accessor` and `Edge` tables are created dynamically at runtime from `LAYER_CONFIG`. This makes the schema highly extensible without boilerplate code.
    -   **Transactional Pattern:** Business logic is encapsulated in `Transaction` classes, separating the user friendly high-level API from the underlying ORM models.
    -   **Layered Graph Model:** The graph is organized into distinct "layers" (e.g., indexing by `tasks`, `topics`, `related_user_request`) which can access the same content but in different ways. Those layers can be queried selectively to combine or isolate properties of different indexing logics.
    -   **Proxy Accessors:** An `Accessor` can act as a proxy for another, enabling aliasing and consolidation of semantic entry points. This is a crucial component for creating "messy" and evolving indexes.
-   **Algorithms & Data Structures**
    -   **NodeSet:** A custom dictionary-like class for aggregating and scoring graph query results. It tracks both a `weight` (e.g., similarity or importance) and a `count` (frequency of appearance), allowing for custom, sophisticated ranking.
    -   **Transitive Weight Calculation:** Content nodes can be searched for by "jumping" between accessor nodes that share edges to a same content node. Nodes retrieved via jumps can be ranked and inserted into other search results according to the product of the weighted edges that were used to access it. Custom search algorithms can be easily implemented.

---

## Highlights

-   **Two-Step Context Retrieval**
    Instead of a direct vector search on raw content, queries first find the most relevant `Accessors` and then traverse their edges to find linked `Content`. This structured approach allows separation of indexing logic and actual content, and the co-existence of multiple semantic indexes of the data.

-   **Configuration-Driven Schema**
    The entire graph structure is defined in `src/database/config.py`. Add a new entry to the `LAYER_CONFIG` dictionary, and the framework automatically generates the required database tables and SQLAlchemy ORM classes on startup.

    ```python
    # src/database/config.py
    "projects": LayerConfig(
        type = "text",
        similarity_condition = l2_norm(0.5),
        embeddings_function = Vectorizer.process,
        embeddings_dimension = 3072,
    ),
    ```

-   **Contextual Retrieval Beyond Simple Vector Search**
    Instead of performing a flat vector search across a sea of raw text, RAGraph uses a more deliberate, two-step process. Queries first identify relevant conceptual entry points (`Accessors`), and then traverse their weighted connections (`Edges`) to find the actual `Content`. Queries are meant to describe a context to search from, not an object similar to the target (allegedly).

-   **Tame Messy, LLM-Generated Knowledge**
    Real-world data is messy, especially when gathered and indexed by a LLM. RAGraph is an experimental way to handle this ambiguity. Multiple indexing layers and proxies provide additional degrees of freedom for indexing data without making the search algorithms too complicated. This freedom can support more elaborate requirements to supply a data gathering LLM. Reducer algorithms can then be used to filter out redundant or irrelevant context by tuning down edge weights or merging accessors (that's a use case for proxies).

-   **Intelligent Result Ranking**
    The search system produces a `NodeSet` that tracks both `weight` (relevance or similarity) and `count` (how many paths led to a node). This allows for sophisticated ranking that can distinguish between a single, highly relevant piece of information and a concept that is moderately relevant but surfaces consistently across many different contexts. This is designed to support custom search algorithm, by making edge data readily available.

---

## Installation

**Prerequisites**
-   Python 3.13+
-   A running PostgreSQL server with the `pgvector` extension enabled.
-   Poetry for dependency management.
-   An Azure OpenAI API key and endpoint.

**Clone and Set Up**

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ryustiel/Ragraph
    cd ragraph
    ```

2.  **Install dependencies:**
    I recommend you use the uv python package manager. All dependencies are in pyproject.toml
    ```bash
    uv sync
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the project root and populate it with your credentials.

    ```bash
    # .env
    DATABASE_CONNECTION_STRING="postgresql+psycopg2://USER:PASSWORD@HOST:PORT/DATABASE?sslmode=require"
    OPENAI_API_KEY=...
    ```

4.  **Initialize the database:**
    The provided `test.py` script includes logic to create the necessary tables and enable the `pgvector` extension. Running it for the first time will set up your database schema.

    ```bash
    uv run python test.py
    ```
    This script serves as a comprehensive example of how to use the library.

---

## Configuration

-   **Environment Variables (`.env`)**
    -   `DATABASE_CONNECTION_STRING`: The full connection string for your PostgreSQL database.
    -   `OPENAI_API_KEY`: Credentials for the OpenAI service used for vectorization.

-   **Database Layers (`src/database/config.py`)**
    -   **`LAYER_CONFIG`**: The main dictionary for defining accessor layers. You can add, remove, or modify layers here. Each layer requires an `embeddings_dimension`, `similarity_condition`, and `embeddings_function`.
    -   **Weight Formulas**: Global functions like `WEIGHT_STRONG_INCREASE` can be tweaked to control how edge weights evolve.

---

## Project Structure

```text
ragraph/
├─ src/
│  ├─ database/
│  │  ├─ config.py              # Core configuration: defines layers, weights, and similarity logic.
│  │  ├─ naming.py              # Functions for generating table and class names dynamically.
│  │  ├─ _A_Edge_Node.py        # Abstract base classes for Node and Edge.
│  │  ├─ _B_Content.py          # ORM model for the Content node.
│  │  ├─ _C_Accessor.py         # Abstract base class for Accessor nodes.
│  │  ├─ _D_Dynamic_Generation.py # Dynamically creates concrete ORM classes from config.py.
│  │  ├─ _database_connection.py# Handles DB connection, session management, and table creation.
│  │  ├─ _node_set.py           # Custom NodeSet class for aggregating weighted results.
│  │  └─ _vectorizer.py         # Wrapper for the OpenAI vectorization client.
│  ├─ transactions/
│  │  ├─ accessor.py            # Pydantic model for Accessor transactions (create, link).
│  │  ├─ content.py             # Pydantic model for Content transactions.
│  │  ├─ context.py             # High-level query interface (Context/AccessorContext).
│  │  └─ node_transaction.py    # Generic base class for transaction models.
│  └─ __init__.py               # Exports key classes for easy import.
├─ test.py                      # Executable example script demonstrating core functionality.
├─ .env                         # Stores environment variables (DB connection, API keys).
└─ pyproject.toml               # Project dependencies and metadata.
```