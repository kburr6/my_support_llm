-- SQL Setup Script for the AI Support Assistant Application
-- Run this script once after creating the database for the first time.

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
-- 