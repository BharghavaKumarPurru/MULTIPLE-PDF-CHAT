import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFaceHub
import logging
import os

# Set up logging to include a file handler
log_file = "app_logs.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),  # Logs to the console
        logging.FileHandler(log_file)  # Logs to a file
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACE_API_TOKEN")
logger.info(f"Hugging Face API key loaded: {os.getenv('HUGGINGFACE_API_TOKEN')[:10]}********")

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    logger.info("Extracting text from PDFs.")
    text = ""
    try:
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        logger.info("Text extraction complete.")
    except Exception as e:
        logger.error(f"Error extracting text from PDFs: {e}")
        raise e
    return text

# Function to split the text into chunks
def get_text_chunks(text):
    logger.info("Splitting text into chunks.")
    try:
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        logger.info(f"Generated {len(chunks)} text chunks.")
    except Exception as e:
        logger.error(f"Error splitting text into chunks: {e}")
        raise e
    return chunks

# Function to create vector store
def get_vectorstore(text_chunks):
    logger.info("Creating vector store with Hugging Face embeddings.")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # Adjust to "cuda" if GPU is available
        )
        vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
        logger.info("Vector store created successfully.")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        raise e
    return vectorstore

# Function to create the conversational chain
def get_conversation_chain(vectorstore):
    logger.info("Creating conversational chain.")
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.3, "max_length": 256}
        )
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        logger.info("Conversational chain created successfully.")
    except Exception as e:
        logger.error(f"Error creating conversational chain: {e}")
        raise e
    return conversation_chain

# Handle user input
def handle_userinput(user_question):
    try:
        logger.info("Handling user input.")
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(f"User: {message.content}")
            else:
                st.write(f"Bot: {message.content}")
        logger.info("User input handled successfully.")
    except Exception as e:
        logger.error(f"Error handling user input: {e}")
        st.error(f"Error: {e}")

# Main Streamlit app
def main():
    logger.info("Starting Streamlit app.")
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                try:
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get the text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vector store
                    vectorstore = get_vectorstore(text_chunks)

                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)
                    st.success("Processing complete! Start chatting with your documents.")
                    logger.info("Processing complete.")
                except Exception as e:
                    logger.error(f"Error during processing: {e}")
                    st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
