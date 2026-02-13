-- Fix ambiguous PostgREST RPC resolution for overloaded match_cached_queries().
-- Run this in Supabase SQL Editor on existing environments.

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

-- Optional cleanup to fully remove ambiguity from the legacy name.
-- Uncomment only if you intentionally want to remove legacy overloads:
-- DROP FUNCTION IF EXISTS match_cached_queries(vector(384), uuid, double precision, integer);
-- DROP FUNCTION IF EXISTS match_cached_queries(vector(384), text, double precision, integer);
