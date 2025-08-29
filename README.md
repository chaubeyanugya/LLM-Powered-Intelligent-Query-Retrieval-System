**Problem Statement**

The project addresses the challenge of building an intelligent query–retrieval system that can process large documents (insurance policies, contracts, HR, and compliance documents) and provide contextual, explainable answers. The system needs to handle natural language queries, perform semantic search and clause matching, and return structured results.

**Solution**

The solution integrates document ingestion, embedding-based semantic search, and large language model reasoning. Documents are split into smaller chunks, embedded using transformer-based models, and stored in a vector database for efficient retrieval. A generative model is then used to provide contextual answers to user queries, grounded in the retrieved clauses. The system is exposed through a secure FastAPI endpoint and deployed on Railway for easy access.

**API Demonstration**

An API demo showcasing the authorization process, available endpoints, and a sample request is attached in this repository as [API Demo (PDF)](API_DEMO.pdf)  
[Sample Output (JSON)](response_1754624975970.json) API_DEMO.pdf.

A sample output from the system is also provided in response_1754624975970.json, which contains the structured answers returned by the API for test queries.

**Tech Stack**

*Backend*: FastAPI

*Vector Database*: Pinecone

*Embeddings*: HuggingFace (all-MiniLM-L6-v2)

*LLM*: Google Gemini (via LangChain)

*Document Processing*: PyPDF, LangChain text splitters

*Deployment*: Railway

*Infrastructure*: Python, Uvicorn, dotenv
