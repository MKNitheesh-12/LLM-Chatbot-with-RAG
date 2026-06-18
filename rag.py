import streamlit as st
from langchain_ollama.llms import OllamaLLM
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Streamlit UI setup
st.title("RAG Chatbot")
st.markdown("Ask any question based on the knowledge base")

# Load and split documents
document_loader = PyMuPDFLoader(r"C:\Users\lumin\Desktop\Local LLM\budget_speech.pdf")  # PDF Loader
documents = document_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(documents)

# Create embeddings and store in ChromaDB
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(split_docs, embedding=embedding_model)
retriever = vector_store.as_retriever()

# Initialize LLM with streaming
llm = OllamaLLM(model="gemma:2b", streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

# Define a prompt for document combination
system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say so. "
    "Keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the combine-documents chain and the retrieval chain
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

# Streamlit Input
query = st.text_input("Enter your question:")
if st.button("Ask"):
    if query:
        with st.spinner("Generating response..."):
            response = rag_chain.invoke({"input": query})
        st.write("### Response:")
        st.write(response["answer"])
