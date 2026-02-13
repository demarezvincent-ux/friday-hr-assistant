-- FRIDAY HR Assistant - Semantic Caching SQL Migration
-- Run this in Supabase SQL Editor to enable query caching
-- This can reduce API calls by ~40% for repeated/similar questions

-- Create the query cache table
CREATE TABLE IF NOT EXISTS query_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    query_embedding vector(384),  -- Matches paraphrase-multilingual-MiniLM-L12-v2
    response TEXT NOT NULL,
    sources JSONB,
    company_id TEXT NOT NULL,  -- TEXT to match your existing schema
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast similarity search
CREATE INDEX IF NOT EXISTS idx_query_cache_embedding ON query_cache 
USING ivfflat (query_embedding vector_cosine_ops) WITH (lists = 50);

-- Index for company filtering
CREATE INDEX IF NOT EXISTS idx_query_cache_company ON query_cache(company_id);

-- Index for TTL cleanup
CREATE INDEX IF NOT EXISTS idx_query_cache_created ON query_cache(created_at);

-- Function to find cached responses by similarity
CREATE OR REPLACE FUNCTION match_cached_queries(
    p_query_embedding vector(384),
    p_company_id TEXT,  -- TEXT to match your existing schema
    p_match_threshold FLOAT DEFAULT 0.95,
    p_match_count INT DEFAULT 1
)
RETURNS TABLE (
    id UUID,
    query_text TEXT,
    response TEXT,
    sources JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    similarity FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT 
        qc.id,
        qc.query_text,
        qc.response,
        qc.sources,
        qc.created_at,
        1 - (qc.query_embedding <=> p_query_embedding) AS similarity
    FROM query_cache qc
    WHERE qc.company_id = p_company_id
      AND 1 - (qc.query_embedding <=> p_query_embedding) >= p_match_threshold
      AND qc.created_at > NOW() - INTERVAL '24 hours'
    ORDER BY similarity DESC
    LIMIT p_match_count;
END;
$$;

-- Canonical non-overloaded function name for PostgREST RPC stability.
-- Use this in application code to avoid ambiguity when legacy overloads exist.
CREATE OR REPLACE FUNCTION match_cached_queries_v2(
    p_query_embedding vector(384),
    p_company_id TEXT,
    p_match_threshold FLOAT DEFAULT 0.95,
    p_match_count INT DEFAULT 1
)
RETURNS TABLE (
    id UUID,
    query_text TEXT,
    response TEXT,
    sources JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    similarity FLOAT
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT 
        qc.id,
        qc.query_text,
        qc.response,
        qc.sources,
        qc.created_at,
        1 - (qc.query_embedding <=> p_query_embedding) AS similarity
    FROM query_cache qc
    WHERE qc.company_id = p_company_id
      AND 1 - (qc.query_embedding <=> p_query_embedding) >= p_match_threshold
      AND qc.created_at > NOW() - INTERVAL '24 hours'
    ORDER BY similarity DESC
    LIMIT p_match_count;
END;
$$;

-- Optional: Function to clean up old cache entries (run periodically)
CREATE OR REPLACE FUNCTION cleanup_query_cache()
RETURNS INTEGER
LANGUAGE plpgsql AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM query_cache 
    WHERE created_at < NOW() - INTERVAL '24 hours';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$;
