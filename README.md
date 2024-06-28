# 10-K Analyzer

This project is designed to analyze Form 10-K financial documents using a local LLM (LlamaCpp) and Streamlit for the user interface.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install -r requirements.txt
```

## Setup

1. **Download the Model:**
   - Ensure you have the GGUF model file and place it in the project directory. Update the `MODEL_PATH` in the code if necessary.
   Note : I personally recommend using the "TheBloke/OpenHermes-2.5-Mistral-7B-GGUF" model on huggingface, as that has provided me with the best output so far. You can view it on hugglingface [here] (https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF) or you can directly download the model [here] (https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q6_K.gguf?download=true)

2. **Prepare PDF Files:**
   - Place your 10-K PDF files in the `data/` directory.

## Running the Application

To start the Streamlit application, run:

```bash
streamlit run local-llm-llamacpp.py
```

## Features

- **PDF Processing & Vector Store Creation:**
  - If the vector store does not exist, the application will process the PDFs and create a vector store.
  - Progress is shown via a progress bar.

- **Chat Interaction:**
  - Users can interact with the assistant by entering questions about the 10-K filings.
  - The assistant will provide responses based on the most relevant chunks from the vector store.

## Code Overview

### Main Components

- **PDF Processing:**
  - Extract text from PDFs and split into chunks.
  - Embed text chunks and create a vector store.

- **Local LLM Initialization:**
  - Configure and initialize the LlamaCpp model.

- **Streamlit UI:**
  - Set up the user interface for interacting with the assistant.

### Key Functions

- `get_pdf_text(pdf_path)`: Extracts text from a PDF file.
- `get_text_chunks(text)`: Splits text into manageable chunks.
- `embed_batch(batch_chunks)`: Embeds a batch of text chunks.
- `create_and_save_vector_store(pdf_dir)`: Processes PDFs and creates a vector store.

### Example Usage

1. **Start the Application:**
   ```bash
   streamlit run local-llm-llamacpp.py
   ```

2. **Interact with the Assistant:**
   - Enter your questions about the 10-K filings in the chat input.
   - The assistant will respond with relevant information extracted from the PDFs.

## Notes

- Ensure the `VECTOR_STORE_FILENAME` is correctly set to the desired filename for the vector store.
- Adjust the `BATCH_SIZE` and other parameters as needed for your specific use case.

For more details, refer to the code in `local-llm-llamacpp.py`.
