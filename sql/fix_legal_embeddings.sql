-- Fix legal_knowledge embeddings: convert TEXT to proper VECTOR type
-- Run this in Supabase SQL Editor

-- First, let's verify the current state
-- SELECT id, pg_typeof(embedding) as current_type FROM legal_knowledge LIMIT 1;

-- The embeddings are stored as TEXT (JSON array strings) but need to be VECTOR(384)
-- Option 1: If the column is already vector but data is text, we need to cast

-- Step 1: Create a temporary column with proper vector type
ALTER TABLE legal_knowledge ADD COLUMN IF NOT EXISTS embedding_new vector(384);

-- Step 2: Copy and cast data from text to vector
-- The text is in format "[-0.08,0.16,...]" which needs to be cast to vector
UPDATE legal_knowledge 
SET embedding_new = CASE 
    WHEN embedding IS NOT NULL AND embedding::text != '' 
    THEN embedding::text::vector(384)
    ELSE NULL 
END
WHERE embedding IS NOT NULL;

-- Step 3: Drop old column and rename new one
ALTER TABLE legal_knowledge DROP COLUMN embedding;
ALTER TABLE legal_knowledge RENAME COLUMN embedding_new TO embedding;

-- Step 4: Recreate the vector index
DROP INDEX IF EXISTS idx_legal_knowledge_embedding;
CREATE INDEX idx_legal_knowledge_embedding 
  ON legal_knowledge USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- Verify the fix
SELECT id, pg_typeof(embedding) as type, 
       (embedding <=> (SELECT embedding FROM legal_knowledge WHERE id = 37 LIMIT 1)) as sample_distance
FROM legal_knowledge 
WHERE embedding IS NOT NULL
LIMIT 5;
