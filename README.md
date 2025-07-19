# my_support_llm
my support llm for class project

The first time you launch, the database will be empty. You need to run your one-time SQL setup commands. Open a new terminal window and run:

docker-compose exec db psql -U user -d vector_db

Enter the password: password

This connects you to the new database container. Paste all your CREATE TABLE, ALTER TABLE, and INSERT commands for the initial setup.

-- SQL Setup Script for the AI Support Assistant Application
-- Run this script once after creating the database for the first time.

-- ====================================================================
-- Step 1: Fix the default LangChain table for advanced JSON queries.
-- This MUST be run before any data is added to the vector store.
-- ====================================================================
ALTER TABLE langchain_pg_embedding ALTER COLUMN cmetadata TYPE jsonb USING cmetadata::text::jsonb;

-- ====================================================================
-- Step 2: Create the application-specific tables.
-- ====================================================================

-- Table for storing "golden" Question-Answer pairs managed by an admin.
CREATE TABLE curated_qa (
    id UUID PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for logging user feedback (thumbs up/down) on AI responses.
CREATE TABLE feedback_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    rating SMALLINT NOT NULL, -- 1 for up, -1 for down
    sources JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for logging the full RAG pipeline for debugging (the "Query Inspector").
CREATE TABLE query_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT,
    question TEXT NOT NULL,
    retrieved_context TEXT,
    final_answer TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Table for storing the dynamic AI configuration settings.
CREATE TABLE ai_configuration (
    id INT PRIMARY KEY,
    system_prompt TEXT NOT NULL,
    score_threshold REAL NOT NULL,
    top_k INT NOT NULL,
    embedding_model TEXT NOT NULL
);

-- ====================================================================
-- Step 3: Insert the initial default configuration into the ai_configuration table.
-- This ensures the application has settings to start with.
-- Note the double single-quote to escape the apostrophe in "don't".
-- ====================================================================
INSERT INTO ai_configuration (id, system_prompt, score_threshold, top_k, embedding_model) VALUES (
    1,
    'You are an expert support agent. Your answers must be accurate, concise, and directly based on the provided context.\nIf the context does not contain the answer to the question, you must state that you don''t have enough information.\nDo not invent information or answer from your general knowledge.',
    0.7,
    5,
    'bge-large-en-v1.5'
);

-- ====================================================================
-- End of Setup Script
-- ====================================================================

Paste it directly into the psql prompt (vector_db=>).

Press Enter. The psql client will execute all the commands in sequence. You should see a series of ALTER TABLE, CREATE TABLE, and INSERT 0 1 messages.

Exit psql by typing \q and pressing Enter.


You also need to pull the LLM models into the new Ollama service. In another new terminal, run:

docker-compose exec ollama ollama pull llama3:8b

Access Your App: Open your web browser and go to http://localhost:8501. Your application will be running, fully containerized and orchestrated.

To rebuild: docker-compose up --build
To shut down: ctrl+c and then docker-compose down

note: bge-large-en-v1.5
