import os
import uuid
import psycopg2
import json
import streamlit as st
import pandas as pd
from datetime import datetime, time
import io

from typing import List

# --- Pydantic v2 Import ---
from pydantic import BaseModel, Field
from typing import List, Union

# --- LangChain Imports (Cleaned Up) ---
from langchain_community.document_loaders import (
    SlackDirectoryLoader, WebBaseLoader, PyPDFLoader, TextLoader, Docx2txtLoader,
    UnstructuredExcelLoader, UnstructuredPowerPointLoader, CSVLoader, UnstructuredMarkdownLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pgvector import PGVector
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document


# --- DATABASE AND MODEL CONFIGURATION ---
DB_CONNECTION = "postgresql+psycopg2://user:password@db:5432/vector_db"  #change from localhost to db for docker
COLLECTION_NAME = "product_docs_collection"
LLM_MODEL = "llama3:8b"
SLACK_IMPORT_BASE_PATH = "C:/Users/kburr/my_support_llm/data"
SUPPORTED_FILE_TYPES = {'pdf', 'txt', 'docx', 'xlsx', 'pptx', 'csv', 'md'}

# --- DATABASE HELPER FUNCTIONS ---

def get_db_connection():
    """Establishes a direct connection to the PostgreSQL database."""
    return psycopg2.connect(
        dbname="vector_db",
        user="user",
        password="password",
        host="db",       #change to db for docker file (localhost otherwise)
        port="5432"
    )

def list_all_documents():
    """
    Lists all unique documents. For Curated Q&A, it shows the question text
    instead of the generic source name.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    
    # This advanced SQL query uses a LEFT JOIN and a CASE statement.
    query = f"""
    SELECT DISTINCT
        emb.cmetadata->>'source_doc_id' as id,
        CASE
            WHEN emb.cmetadata->>'source_doc_name' = 'Curated Q&A'
            THEN 'Q: ' || cqa.question -- If it's a Q&A, show the question text
            ELSE emb.cmetadata->>'source_doc_name' -- Otherwise, show the filename
        END as name
    FROM
        langchain_pg_embedding as emb
    LEFT JOIN
        curated_qa as cqa ON (emb.cmetadata->>'source_doc_id')::uuid = cqa.id
    WHERE
        emb.collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = '{COLLECTION_NAME}');
    """
    
    cur.execute(query)
    # The order of columns in the result is now id, name
    documents = cur.fetchall()
    cur.close()
    conn.close()
    
    # We switched the order of id and name in the SELECT, so adjust the unpacking here
    return [{"id": doc[0], "name": doc[1]} for doc in documents]


def delete_document_by_id(doc_id: str):
    """
    Deletes a document and all its chunks from the vector store.
    This version has the correct type casting for the parameterized query,
    and assumes the cmetadata column has been altered to be of type JSONB.
    """
    conn = None
    deleted_count = 0
    
    print("--- Attempting to delete document ---")
    print(f"  - Collection Name Target: '{COLLECTION_NAME}'")
    print(f"  - Document ID to Delete: '{doc_id}'")

    try:
        conn = get_db_connection()
        cur = conn.cursor()

        metadata_to_find = {"source_doc_id": doc_id}
        
        # This query now works because the cmetadata column is jsonb
        query = f"""
            DELETE FROM langchain_pg_embedding
            WHERE collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = %s)
            AND cmetadata @> %s::jsonb;
        """

        cur.execute(query, (COLLECTION_NAME, json.dumps(metadata_to_find)))
        
        deleted_count = cur.rowcount
        conn.commit()
        
        print(f"  - Database command executed successfully.")
        print(f"  - Rows deleted: {deleted_count}")
        print("---------------------------------------")

    except Exception as e:
        print(f"  - ERROR: An exception occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

    return deleted_count

# --- CORE RAG LOGIC ---

# Memoized function to load embeddings only once
@st.cache_resource
def get_embeddings():
    """Loads the embedding model from the local path inside the Docker image."""
    config = get_ai_config()
    model_name = config.get('embedding_model', 'BAAI/bge-large-en-v1.5')
    
    # We now specify the `cache_folder` to point to the directory where
    # our Dockerfile downloaded the model.
    model_kwargs = {'device': 'cpu'} # Use CPU for embeddings in the container
    encode_kwargs = {'normalize_embeddings': True}
    
    print(f"Loading embedding model '{model_name}' from local models directory...")
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder="./models", # Point to the local folder in the container
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

def get_ai_config():
    """Retrieves the current AI configuration from the database."""
    with get_db_connection() as conn:
        # Make sure to select the new embedding_model column
        query = "SELECT system_prompt, score_threshold, top_k, embedding_model FROM ai_configuration WHERE id = 1"
        df = pd.read_sql(query, conn)
        if df.empty:
            raise ValueError("AI configuration not found in the database.")
        return df.iloc[0]

def update_ai_config(system_prompt: str, score_threshold: float, top_k: int, embedding_model: str):
    """Updates the AI configuration in the database."""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Make sure to update the new embedding_model column
            query = """
                UPDATE ai_configuration
                SET system_prompt = %s, score_threshold = %s, top_k = %s, embedding_model = %s
                WHERE id = 1
            """
            cur.execute(query, (system_prompt, score_threshold, top_k, embedding_model))
    print("Successfully updated AI configuration.")
    
# Memoized function to get vector store
@st.cache_resource
def get_vector_store():
    """Gets a memoized instance of the vector store."""
    print("Initializing vector store connection...")
    return PGVector(connection_string=DB_CONNECTION, embedding_function=get_embeddings(), collection_name=COLLECTION_NAME)

def add_document_to_db(file_path: str, file_type: str):
    """Loads, splits, and adds a new document to the vector store."""
    print(f"Adding document: {file_path} of type: {file_type}")
    
    if file_type == 'pdf':
        loader = PyPDFLoader(file_path)
    elif file_type == 'txt':
        loader = TextLoader(file_path)
    # --- START OF NEW CODE ---
    elif file_type == 'docx':
        loader = Docx2txtLoader(file_path)
    elif file_type == 'xlsx':
        # mode="elements" is great for Excel as it treats cells and tables as distinct pieces
        loader = UnstructuredExcelLoader(file_path, mode="elements")
    elif file_type == 'pptx':
        # mode="elements" is also good for PowerPoint
        loader = UnstructuredPowerPointLoader(file_path, mode="elements")
    elif file_type == 'csv':
        loader = CSVLoader(file_path) # You can specify a source_column if needed
    elif file_type == 'md':
        loader = UnstructuredMarkdownLoader(file_path)
    # --- END OF NEW CODE ---
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    documents = loader.load()
    doc_id, doc_name = str(uuid.uuid4()), os.path.basename(file_path)
    
    for doc in documents:
        # Ensure metadata is consistently structured
        if "metadata" not in doc:
            doc.metadata = {}
        doc.metadata["source_doc_id"] = doc_id
        doc.metadata["source_doc_name"] = doc_name
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    doc_chunks = text_splitter.split_documents(documents)
    
    db = get_vector_store()
    db.add_documents(doc_chunks)
    print(f"Successfully added '{doc_name}' with ID: {doc_id}")
    
def add_directory_to_db(dir_path: str):
    """
    Recursively scans a directory, finds all supported files,
    and ingests them into the vector store.
    """
    processed_files = []
    failed_files = []

    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"The specified directory does not exist: {dir_path}")

    # os.walk is a generator that yields the path, subdirectories, and files for each level
    for root, dirs, files in os.walk(dir_path):
        for file_name in files:
            # Check if the file extension is in our set of supported types
            file_ext = file_name.split('.')[-1].lower()
            if file_ext in SUPPORTED_FILE_TYPES:
                full_path = os.path.join(root, file_name)
                try:
                    # We can reuse our existing function for single-file processing!
                    add_document_to_db(full_path, file_ext)
                    processed_files.append(file_name)
                except Exception as e:
                    print(f"Failed to process file {full_path}: {e}")
                    failed_files.append(f"{file_name} (Error: {e})")

    return processed_files, failed_files
    
def add_url_to_db(url: str):
    """Fetches content from a URL, loads it, and adds it to the vector store."""
    print(f"Adding URL: {url}")
    
    # Use WebBaseLoader to fetch and parse the content from the URL
    # It automatically handles fetching the HTML and extracting the main text
    loader = WebBaseLoader(url)
    documents = loader.load()
    
    # We use a UUID for the ID and the URL itself as the name
    doc_id, doc_name = str(uuid.uuid4()), url
    
    for doc in documents:
        if "metadata" not in doc:
            doc.metadata = {}
        doc.metadata["source_doc_id"] = doc_id
        # Use the URL as the source name for easy identification
        doc.metadata["source_doc_name"] = doc_name
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    doc_chunks = text_splitter.split_documents(documents)
    
    db = get_vector_store()
    db.add_documents(doc_chunks)
    print(f"Successfully added content from '{doc_name}' with ID: {doc_id}")

def add_slack_dir_to_db(dir_path: str):
    """Loads all conversations from a Slack export directory and adds them to the vector store."""
    print(f"Adding Slack export from directory: {dir_path}")

    # Check if the directory exists to provide a better error message
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"The specified directory does not exist: {dir_path}")

    # The loader recursively finds all .json files in the directory structure
    loader = SlackDirectoryLoader(dir_path)
    documents = loader.load()

    # The "source" metadata from this loader is very useful, as it includes the channel and timestamp.
    # We will enrich it with our own doc_id for deletion purposes.
    doc_id = str(uuid.uuid4())
    # We'll use the directory path as the "document name"
    doc_name = f"Slack Export: {os.path.basename(dir_path)}"

    for doc in documents:
        if "metadata" not in doc:
            doc.metadata = {}
        # Add our custom metadata for tracking and deletion
        doc.metadata["source_doc_id"] = doc_id
        doc.metadata["source_doc_name"] = doc_name
        # The loader already adds a 'source' key (e.g., 'my-channel/2024-01-01.json'),
        # which is great, so we'll leave it.

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    doc_chunks = text_splitter.split_documents(documents)

    db = get_vector_store()
    db.add_documents(doc_chunks)
    print(f"Successfully added content from '{doc_name}' with ID: {doc_id}")

@st.cache_resource
def get_chains():
    """
    Creates and returns the retriever and generation chain, using settings
    fetched from the ai_configuration table and a proper chat prompt structure.
    """
    print("Fetching AI config and creating retriever and generation chain...")
    
    config = get_ai_config()
    system_prompt_text = config['system_prompt'] # The rules for the AI
    score_threshold = config['score_threshold']
    top_k = config['top_k']
    
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k, "score_threshold": score_threshold}
    )
    
    llm = Ollama(model=LLM_MODEL, base_url="http://ollama:11434")       #change to for docker file: llm = Ollama(model=LLM_MODEL, host="http://ollama:11434")

    # --- START OF THE CRITICAL FIX ---
    
    # 1. Define the structure for the "human" message part of the prompt.
    # This part contains the data (context and question) that changes with each query.
    human_prompt_template = HumanMessagePromptTemplate.from_template(
        """
        Here is the context to use for answering the question:
        ---
        CONTEXT:
        {context}
        ---
        
        Here is the user's question:
        
        QUESTION:
        {question}
        """
    )
    
    # 2. Create the ChatPromptTemplate from a list of message templates.
    # This clearly separates the unchanging system instructions from the dynamic user input.
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_prompt_text),
        human_prompt_template
    ])

    # Create the generation chain using the new, correctly structured chat_prompt
    generation_chain = (
        chat_prompt
        | llm
        | StrOutputParser()
    )
    
    return retriever, generation_chain

import pandas as pd
from langchain_core.documents import Document

def get_all_curated_qa():
    """
    Retrieves all curated Q&A pairs and joins them with feedback counts
    from the feedback_log table.
    """
    conn = get_db_connection()
    
    # This SQL query does the heavy lifting:
    # 1. LEFT JOINs curated_qa with feedback_log on the question text.
    # 2. Counts thumbs up (rating=1) and thumbs down (rating=-1).
    # 3. Uses COALESCE to ensure counts are 0 instead of NULL if no feedback exists.
    # 4. Groups the results by each unique Q&A pair.
    query = """
    SELECT
        cqa.id,
        cqa.question,
        cqa.answer,
        cqa.created_at,
        COALESCE(SUM(CASE WHEN fl.rating = 1 THEN 1 ELSE 0 END), 0) AS thumbs_up,
        COALESCE(SUM(CASE WHEN fl.rating = -1 THEN 1 ELSE 0 END), 0) AS thumbs_down
    FROM
        curated_qa AS cqa
    LEFT JOIN
        feedback_log AS fl ON cqa.question = fl.question
    GROUP BY
        cqa.id, cqa.question, cqa.answer, cqa.created_at
    ORDER BY
        cqa.created_at DESC;
    """
    
    # Pandas reads the result directly into a DataFrame with the new columns
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Convert counts to integers as they might come back as a different numeric type
    df['thumbs_up'] = df['thumbs_up'].astype(int)
    df['thumbs_down'] = df['thumbs_down'].astype(int)
    
    return df

def add_curated_qa(question: str, answer: str):
    """Adds a new Q&A pair to both the SQL table and the vector store."""
    qa_id = uuid.uuid4()
    
    # 1. Add to the standard SQL table for management
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO curated_qa (id, question, answer) VALUES (%s, %s, %s)",
        (str(qa_id), question, answer)
    )
    conn.commit()
    cur.close()
    conn.close()

    # 2. Add to the vector store for retrieval by the LLM
    # We format it as a single document for better semantic meaning
    text_content = f"Question: {question}\nAnswer: {answer}"
    
    # Create a LangChain Document object with specific metadata
    doc = Document(
        page_content=text_content,
        metadata={
            "source_doc_id": str(qa_id), # Use the same ID for easy linking
            "source_doc_name": "Curated Q&A"
        }
    )
    
    vector_store = get_vector_store()
    vector_store.add_documents([doc])
    print(f"Successfully added curated Q&A with ID: {qa_id}")

def delete_curated_qa(qa_id: str):
    """
    Deletes a Q&A pair from the SQL table, its associated feedback from the
    feedback_log, and its chunks from the vector store.
    """
    conn = None
    question_to_delete = None
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # STEP 1: First, get the question text before we delete the record.
        # We need this to find and delete the associated feedback.
        cur.execute("SELECT question FROM curated_qa WHERE id = %s", (qa_id,))
        result = cur.fetchone()
        if result:
            question_to_delete = result[0]
        else:
            print(f"WARNING: No curated Q&A found with ID {qa_id}. Cannot proceed with delete.")
            return

        # STEP 2: Delete the entry from the curated_qa table.
        cur.execute("DELETE FROM curated_qa WHERE id = %s", (qa_id,))
        print(f"Deleted Q&A pair with ID {qa_id} from curated_qa table.")

        # STEP 3: If we found a question, delete all associated feedback.
        if question_to_delete:
            cur.execute("DELETE FROM feedback_log WHERE question = %s", (question_to_delete,))
            # .rowcount tells us how many feedback records were deleted.
            feedback_deleted_count = cur.rowcount
            print(f"Deleted {feedback_deleted_count} associated records from feedback_log.")

        conn.commit() # Commit all delete operations at once.

    except Exception as e:
        print(f"ERROR deleting curated Q&A: {e}")
        if conn:
            conn.rollback()
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
            
    # STEP 4: Finally, delete the chunks from the vector store.
    # This is done outside the transaction as it's a separate logical unit.
    if question_to_delete:
        vector_chunks_deleted = delete_document_by_id(qa_id)
        print(f"Deleted {vector_chunks_deleted} vector store chunks for Q&A ID {qa_id}.")

# NOTE: We are intentionally not implementing an "edit" function for simplicity.
# The user can achieve an edit by deleting the old entry and adding a new one.
# A full edit-in-place would require deleting the old vector embedding and adding a new one.

def log_feedback(session_id: str, question: str, answer: str, sources: list, rating: int):
    """Logs user feedback to the feedback_log table."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Sanitize the sources to a simpler format for JSON storage
        # This converts the complex Document objects into a list of dictionaries
        simplified_sources = [
            {"page_content": doc.page_content, "metadata": doc.metadata} for doc in sources
        ]
        
        # Convert the list of dictionaries to a JSON string
        sources_json = json.dumps(simplified_sources, indent=2)
        
        query = """
            INSERT INTO feedback_log (session_id, question, answer, rating, sources)
            VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(query, (session_id, question, answer, rating, sources_json))
        conn.commit()
        print(f"Successfully logged feedback with rating: {rating}")
    except Exception as e:
        print(f"ERROR logging feedback: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            
def get_feedback_log():
    """Retrieves all data from the feedback_log table."""
    conn = get_db_connection()
    
    # Simple query to get all feedback, ordered by most recent first
    query = """
    SELECT
        created_at,
        rating,
        question,
        answer,
        sources,
        session_id
    FROM
        feedback_log
    ORDER BY
        created_at DESC;
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

def log_query(session_id: str, question: str, context: str, answer: str):
    """Logs the entire RAG query flow to the query_log table."""
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # This query now correctly includes the 'retrieved_context' column
        query = """
            INSERT INTO query_log (session_id, question, retrieved_context, final_answer)
            VALUES (%s, %s, %s, %s)
        """
        
        # This execute call now correctly passes all four arguments
        cur.execute(query, (session_id, question, context, answer))
        
        conn.commit()
        print("Successfully logged query flow with context.")

    except Exception as e:
        print(f"ERROR logging query: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

def get_query_log(start_date=None, end_date=None, search_term=None):
    """
    Retrieves data from the query_log table, with optional filters for
    date range and a text search term.
    """
    conn = get_db_connection()
    
    # Base query
    base_query = "SELECT created_at, question, retrieved_context, final_answer, session_id FROM query_log"
    
    # Lists to hold WHERE clauses and parameters for safe querying
    where_clauses = []
    params = []
    
    # Add date range filters if provided
    if start_date:
        where_clauses.append("created_at >= %s")
        params.append(start_date)
    if end_date:
        # To make the end date inclusive, we look for dates before the start of the next day
        end_of_day = datetime.combine(end_date, time.max)
        where_clauses.append("created_at <= %s")
        params.append(end_of_day)

    # Add text search filter if provided
    if search_term:
        # Case-insensitive search across question and answer columns
        where_clauses.append("(question ILIKE %s OR final_answer ILIKE %s)")
        # Add the parameter twice, once for each column
        search_pattern = f"%{search_term}%"
        params.append(search_pattern)
        params.append(search_pattern)

    # Combine all parts into a final query
    if where_clauses:
        final_query = f"{base_query} WHERE {' AND '.join(where_clauses)} ORDER BY created_at DESC"
    else:
        final_query = f"{base_query} ORDER BY created_at DESC"
        
    print(f"Executing query log search: {final_query}")
    
    # Use pandas to execute the parameterized query
    df = pd.read_sql(final_query, conn, params=tuple(params))
    conn.close()
    
    return df

def get_ai_config():
    """Retrieves the current AI configuration from the database."""
    conn = get_db_connection()
    # This query now correctly selects the new embedding_model column
    query = "SELECT system_prompt, score_threshold, top_k, embedding_model FROM ai_configuration WHERE id = 1"
    df = pd.read_sql(query, conn)
    conn.close()
    if df.empty:
        raise ValueError("AI configuration not found in the database. Please insert the default row.")
    # Return the first (and only) row as a dictionary-like object
    return df.iloc[0]

def update_ai_config(system_prompt: str, score_threshold: float, top_k: int, embedding_model: str):
    """Updates the AI configuration in the database."""
    conn = get_db_connection()
    cur = conn.cursor()
    # This query now correctly updates the new embedding_model column
    query = """
        UPDATE ai_configuration
        SET system_prompt = %s, score_threshold = %s, top_k = %s, embedding_model = %s
        WHERE id = 1
    """
    # The parameters tuple now correctly includes the 4th argument
    cur.execute(query, (system_prompt, score_threshold, top_k, embedding_model))
    conn.commit()
    cur.close()
    conn.close()
    print("Successfully updated AI configuration.")
    
def get_dashboard_kpis():
    """Calculates key performance indicators for the dashboard."""
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Get total queries
    cur.execute("SELECT COUNT(*) FROM query_log")
    total_queries = cur.fetchone()[0]
    
    # Get total feedback entries
    cur.execute("SELECT COUNT(*) FROM feedback_log")
    total_feedback = cur.fetchone()[0]
    
    # Get positive feedback count
    cur.execute("SELECT COUNT(*) FROM feedback_log WHERE rating = 1")
    positive_feedback = cur.fetchone()[0]
    
    # Get negative feedback count
    cur.execute("SELECT COUNT(*) FROM feedback_log WHERE rating = -1")
    negative_feedback = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    return {
        "total_queries": total_queries,
        "total_feedback": total_feedback,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
    }

def get_top_problem_questions(limit=10):
    """Finds the questions that have received the most downvotes."""
    conn = get_db_connection()
    query = """
        SELECT question, COUNT(*) as downvote_count
        FROM feedback_log
        WHERE rating = -1
        GROUP BY question
        ORDER BY downvote_count DESC
        LIMIT %s
    """
    df = pd.read_sql(query, conn, params=(limit,))
    conn.close()
    return df

def get_top_information_gaps(limit=10):
    """Finds questions where the bot frequently answered 'I don't know'."""
    conn = get_db_connection()
    query = """
        SELECT question, COUNT(*) as failure_count
        FROM query_log
        WHERE final_answer ILIKE '%%enough information%%'
        GROUP BY question
        ORDER BY failure_count DESC
        LIMIT %s
    """
    df = pd.read_sql(query, conn, params=(limit,))
    conn.close()
    return df

def get_most_frequent_questions(limit=10):
    """Finds the most frequently asked questions overall."""
    conn = get_db_connection()
    query = """
        SELECT question, COUNT(*) as query_count
        FROM query_log
        GROUP BY question
        ORDER BY query_count DESC
        LIMIT %s
    """
    df = pd.read_sql(query, conn, params=(limit,))
    conn.close()
    return df

def reset_logs_for_question(question: str):
    """
    Deletes all entries from feedback_log and query_log that match
    a specific question string.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Delete from feedback_log
        cur.execute("DELETE FROM feedback_log WHERE question = %s", (question,))
        feedback_deleted = cur.rowcount
        print(f"Reset feedback log: Deleted {feedback_deleted} entries for question '{question}'")

        # Delete from query_log
        cur.execute("DELETE FROM query_log WHERE question = %s", (question,))
        query_deleted = cur.rowcount
        print(f"Reset query log: Deleted {query_deleted} entries for question '{question}'")

        conn.commit()
        return feedback_deleted, query_deleted

    except Exception as e:
        print(f"ERROR resetting logs for question: {e}")
        if conn:
            conn.rollback()
        return 0, 0
    finally:
        if conn:
            conn.close()

class PlottingToolInput(BaseModel):
    code: str = Field(description="The Python code to execute for generating a plot.")


def delete_all_data():
    """
    Deletes ALL data from the knowledge base. This includes all vector embeddings,
    all curated Q&A pairs, and all associated logs. This is a destructive operation.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # 1. Truncate vector store tables. TRUNCATE is faster than DELETE for full tables.
        # LangChain creates these two tables for PGVector.
        cur.execute(f"TRUNCATE TABLE langchain_pg_embedding, langchain_pg_collection RESTART IDENTITY;")
        print("Successfully truncated vector store tables.")

        # 2. Truncate the curated Q&A table.
        cur.execute("TRUNCATE TABLE curated_qa RESTART IDENTITY;")
        print("Successfully truncated curated_qa table.")
        
        # 3. Truncate the log tables.
        cur.execute("TRUNCATE TABLE feedback_log RESTART IDENTITY;")
        cur.execute("TRUNCATE TABLE query_log RESTART IDENTITY;")
        print("Successfully truncated log tables.")

        conn.commit()
        print("All knowledge base data has been successfully deleted.")
        return True

    except Exception as e:
        print(f"ERROR during 'delete_all_data': {e}")
        if conn:
            conn.rollback()
        return False
    finally:
        if conn:
            conn.close()