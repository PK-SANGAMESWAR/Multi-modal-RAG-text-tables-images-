# Multimodal RAG

This project implements a Multimodal Retrieval-Augmented Generation (RAG) system capable of processing and retrieving information from documents containing text, tables, and images.

## Features

- **PDF Partitioning**: Uses `unstructured` to parse PDFs into text, tables, and images.
- **Hierarchical Chunking**: Chunks text by title for better context preservation.
- **Multimodal Embeddings/Summarization**:
  - Uses a local LLM (Ollama with `llama3.2:3b`) to generate summaries for images and tables.
  - Stores these summaries for semantic retrieval.
- **Vector Database**: Uses `Chroma` for storing and retrieving document chunks.
- **LangChain Integration**: Built using `langchain`, `langchain-ollama`, and `langchain-chroma`.

## Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) installed and running.
- Pull the required model:
  ```bash
  ollama pull llama3.2:3b
  ```

## Installation

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `unstructured[all-docs]` requires system dependencies like `poppler-utils` and `tesseract-ocr` may need to be installed separately depending on your OS.*

## Usage

1. Ensure your `.env` file is configured if necessary (though this project primarily uses local Ollama).
2. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook multi_moda_rag.ipynb
   ```
3. The notebook guides you through:
   - Partioning a PDF.
   - Summarizing tables and images using the local LLM.
   - Indexing the content into Chroma.
   - Querying the RAG system.

## Project Structure

- `multi_moda_rag.ipynb`: Main notebook containing the RAG implementation.
- `requirements.txt`: Python dependencies.
- `docs/`: Directory for input documents (e.g., PDFs).
- `db/`: Directory for Chroma vector database storage.
