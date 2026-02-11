-- ============================================
-- Legal Hierarchy Migration
-- Adds legal_tier column and tier-aware RPC
-- Run this in Supabase SQL Editor
-- ============================================

-- Step 1: Add legal_tier column (1=law, 2=sector, 3=company)
ALTER TABLE legal_knowledge
  ADD COLUMN IF NOT EXISTS legal_tier smallint NOT NULL DEFAULT 3;

COMMENT ON COLUMN legal_knowledge.legal_tier IS
  '1=Federal law (highest authority), 2=Sector CAO/CCT, 3=Company policy (lowest)';

-- Step 2: Index for tier-based queries
CREATE INDEX IF NOT EXISTS idx_legal_knowledge_tier
  ON legal_knowledge (legal_tier);

-- Step 3: Backfill existing rows from metadata->>'source'
UPDATE legal_knowledge SET legal_tier = 1
  WHERE metadata->>'source' = 'BELGIAN_LAW'
    OR metadata->>'source' = 'FOUNDATION_LAW'
    OR metadata->>'category' = 'legal_foundation';

UPDATE legal_knowledge SET legal_tier = 2
  WHERE legal_tier = 3  -- only update rows not already set
    AND (
      metadata->>'source' LIKE 'CAO%'
      OR metadata->>'source' LIKE 'PC%'
      OR metadata->>'source' LIKE 'CCT%'
      OR metadata->>'source' = 'NAR'
      OR metadata->>'source' = 'CNT'
    );

-- Rows that are already 3 (default) stay as company policy — no action needed.

-- Step 4: Tier-aware hybrid search RPC
CREATE OR REPLACE FUNCTION match_legal_documents_tiered(
  query_embedding vector(384),
  text_search_query text,
  match_threshold float DEFAULT 0.15,
  match_count int DEFAULT 10,
  source_filter text DEFAULT NULL,
  tier_filter smallint DEFAULT NULL   -- NULL = all tiers
)
RETURNS TABLE (
  id bigint,
  content text,
  summary text,
  metadata jsonb,
  legal_tier smallint,
  similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY

  WITH vector_candidates AS (
    SELECT
      lk.id,
      lk.content,
      lk.summary,
      lk.metadata,
      lk.legal_tier,
      -- Boost vector similarity into 0.4–1.0 range
      0.4 + (0.6 * (1 - (lk.embedding <=> query_embedding))) AS similarity
    FROM legal_knowledge lk
    WHERE (source_filter IS NULL OR (lk.metadata->>'source') = source_filter)
      AND (tier_filter IS NULL OR lk.legal_tier = tier_filter)
      AND lk.embedding IS NOT NULL
      AND 1 - (lk.embedding <=> query_embedding) > match_threshold
    ORDER BY lk.embedding <=> query_embedding
    LIMIT match_count
  ),

  keyword_candidates AS (
    SELECT
      lk.id,
      lk.content,
      lk.summary,
      lk.metadata,
      lk.legal_tier,
      0.3::float AS similarity
    FROM legal_knowledge lk
    WHERE (source_filter IS NULL OR (lk.metadata->>'source') = source_filter)
      AND (tier_filter IS NULL OR lk.legal_tier = tier_filter)
      AND length(text_search_query) > 0
      AND to_tsvector('simple', lk.content) @@ to_tsquery('simple', text_search_query)
    LIMIT match_count
  ),

  combined AS (
    SELECT * FROM vector_candidates
    UNION ALL
    SELECT * FROM keyword_candidates
  ),

  deduplicated AS (
    SELECT DISTINCT ON (c.id)
      c.id,
      c.content,
      c.summary,
      c.metadata,
      c.legal_tier,
      c.similarity
    FROM combined c
    ORDER BY c.id, c.similarity DESC
  )

  -- Final sort: tier first (law before sector before company), then similarity
  SELECT * FROM deduplicated
  ORDER BY deduplicated.legal_tier ASC, deduplicated.similarity DESC
  LIMIT match_count;

END;
$$;

-- Step 5: Grant permissions
GRANT EXECUTE ON FUNCTION match_legal_documents_tiered TO anon, authenticated;

-- Verify migration
SELECT legal_tier, COUNT(*) as count
FROM legal_knowledge
GROUP BY legal_tier
ORDER BY legal_tier;
