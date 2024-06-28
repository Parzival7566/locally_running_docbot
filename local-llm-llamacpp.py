import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_community.llms import LlamaCpp
import math 
import re

# --- Constants ---
PDF_DIRECTORY = "data/"  # Directory containing your 10-K PDFs
VECTOR_STORE_FILENAME = "faiss_index"
BATCH_SIZE = 100  # Batch size for embeddings
MODEL_PATH = "openhermes-2.5-mistral-7b.Q6_K.gguf"  # Path to your GGUF model file

# --- Initialize Vector Store ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = None

# --- Initialize Local LLM ---
llm = LlamaCpp(
    model_path=MODEL_PATH,
    n_ctx=4096,  # Increased context window
    n_batch=512,
    n_gpu_layers=-1, # Use all available GPU layers, 0 for CPU
    temperature=0.1,  # Lowered temperature for more focused outputs
    max_tokens=512,  # Increased max tokens
    verbose=True,
    use_mlock=True,
    use_mmap=True,
)

# --- Functions ---
def get_pdf_text(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

async def embed_batch(batch_chunks):
    return embeddings.embed_documents(batch_chunks)

async def create_and_save_vector_store(pdf_dir):
    global vector_store
    all_chunks = []

    # --- 1. Get all chunks from PDF files ---
    for i, filename in enumerate(os.listdir(pdf_dir)):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            text = get_pdf_text(pdf_path)
            chunks = get_text_chunks(text)
            all_chunks.extend(chunks)

            # Update progress bar for files
            progress_bar.progress((i + 1) / len(os.listdir(pdf_dir)), 
                                  text=f"Processing PDF {i+1}/{len(os.listdir(pdf_dir))}")

    # --- 2. Batch Embedding ---
    num_batches = math.ceil(len(all_chunks) / BATCH_SIZE)
    embedding_tasks = []
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = min((batch_num + 1) * BATCH_SIZE, len(all_chunks))
        batch_chunks = all_chunks[start_idx:end_idx]
        embedding_tasks.append(embed_batch(batch_chunks))

        # Update progress bar for batches 
        progress_bar.progress((batch_num + 1) / num_batches, 
                              text=f"Embedding batch {batch_num+1}/{num_batches}")

    # --- 3. Gather Embedding Results ---
    embeddings_list = await asyncio.gather(*embedding_tasks)
    embeddings_list = [embedding for batch in embeddings_list for embedding in batch]  # Flatten the list

    # --- 4. Create FAISS Index ---
    text_embeddings = list(zip(all_chunks, embeddings_list))
    vector_store = FAISS.from_embeddings(text_embeddings, embeddings)
    vector_store.save_local(VECTOR_STORE_FILENAME)

def load_vector_store():
    global vector_store
    vector_store = FAISS.load_local(VECTOR_STORE_FILENAME, embeddings, allow_dangerous_deserialization=True)


def get_conversational_chain():
    prompt_template = """
    You are a helpful AI assistant designed to answer questions about financial documents of different companies. 
    You have been provided with information from Form 10-K filings of multiple companies.
    
    Use the provided context to answer the question as accurately as possible. 
    Whenever the user asks a question about google, answer the question with the context of the alphabet document as google is a subsidary of alphabet; provide a disclaimer for this as well.
    If the question asks for a comparison, make sure to highlight the differences and similarities between the companies.
    If you cannot answer the question from the given context, say "I'm sorry, I don't have enough information to answer that."
    
    Always start your answer with the company name(s) relevant to the question.
    If asked about specific financial figures, provide the exact numbers from the context if available.
    
    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# --- Streamlit UI ---
st.set_page_config(page_title="10-K Analyzer", page_icon="📈")
st.title("🔍 Analyze Financial Documents")

# --- Session State Initialization ---
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I can help analyze Form 10-K documents. Ask me anything! 😊"}
    ]

# --- PDF Processing & Vector Store ---
if not os.path.exists(VECTOR_STORE_FILENAME):
    with st.spinner("Processing PDFs..."):
        progress_bar = st.progress(0, text="Starting...")
        asyncio.run(create_and_save_vector_store(PDF_DIRECTORY)) 
        progress_bar.progress(1.0, text="PDFs processed and vector store created!")
        st.success("PDFs processed and vector store created!")

if os.path.exists(VECTOR_STORE_FILENAME) and vector_store is None:
    with st.spinner("Loading vector store..."):
        load_vector_store()

# --- Chat Interaction ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Enter your question about the 10-K filings..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            try:
                docs = vector_store.similarity_search(prompt, k=5)  # Increased to top 5 relevant chunks
                chain = get_conversational_chain()
                response = chain.invoke({"input_documents": docs, "question": prompt}, return_only_outputs=True)
                st.write(response["output_text"])
                st.session_state.messages.append({"role": "assistant", "content": response["output_text"]})
            except Exception as e:
                st.error(f"An error occurred while processing your request: {str(e)}")
