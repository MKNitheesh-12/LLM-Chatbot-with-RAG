# LLM Chatbot with RAG

A simple Retrieval-Augmented Generation (RAG) chatbot built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), and a locally-hosted [Ollama](https://ollama.com/) LLM. The chatbot answers questions based on the content of a PDF document, using vector search to retrieve relevant context before generating a response.

This app runs entirely on your local machine — no external API keys or cloud LLM calls required.

## How It Works

1. A PDF document is loaded and split into overlapping text chunks.
2. Each chunk is embedded using a HuggingFace sentence-transformer model.
3. The embeddings are stored in a local ChromaDB vector store.
4. When you ask a question, the most relevant chunks are retrieved and passed as context to the LLM.
5. The LLM (Mistral 7B Instruct, running locally via Ollama) generates an answer based on that context.

## Tech Stack

- **Streamlit** – web UI
- **LangChain** – orchestration of the retrieval and generation pipeline
- **ChromaDB** – local vector store
- **HuggingFace Embeddings** – `sentence-transformers/all-MiniLM-L6-v2`
- **Ollama** – local LLM runtime
- **Mistral 7B Instruct** (`mistral:7b-instruct-q4_K_M`) – the language model used in `mistral.py`

## Prerequisites

- Python 3.9+
- [Ollama](https://ollama.com/download) installed and running locally
- The Mistral model pulled in Ollama:
  ```bash
  ollama pull mistral:7b-instruct-q4_K_M
  ```

## Installation

1. Clone the repository:
   ```bash
   git clone <your-repo-url>
   cd LLM-Chatbot-with-RAG-master
   ```

2. Install the required Python packages:
   ```bash
   pip install streamlit langchain langchain-ollama langchain-chroma langchain-huggingface langchain-community pymupdf sentence-transformers
   ```

## Setup

The script loads a PDF from a hardcoded local file path. Before running it:

1. Place your PDF file (e.g. `budget_speech.pdf`) somewhere on your machine.
2. Open `mistral.py` and update this line with the correct path to your PDF:
   ```python
   document_loader = PyMuPDFLoader(r"C:\Users\lumin\Desktop\Local LLM\budget_speech.pdf")
   ```

## Usage

Make sure Ollama is running, then start the app:

```bash
streamlit run mistral.py
```

This will open the chatbot in your browser. Type a question in the input box and click **Ask** to get an answer based on the contents of your PDF.

## Notes

- On first run, the app will download the embedding model and build the vector store from the PDF, which may take a moment depending on document size.
- Responses are also streamed to the terminal/console in addition to the Streamlit UI.
- To use a different LLM, change the `model` argument in the `OllamaLLM(...)` call to any model you've pulled via Ollama.

## License

This project currently has no license specified. Add one if you intend to share or open-source this code.
