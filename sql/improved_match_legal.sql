-- Fix match_legal_documents to properly weight vector vs FTS results
-- Run this in Supabase SQL Editor

-- The issue: FTS results get a flat 0.5 score which often beats vector similarity
-- Fix: Use weighted hybrid scoring where vector gets boosted

create or replace function match_legal_documents(
  query_embedding vector(384),
  text_search_query text,      -- Pipe-delimited: 'term1 | term2'
  match_threshold float default 0.15,
  match_count int default 10,
  source_filter text default null  -- Filter by source like 'BELGIAN_LAW'
)
returns table (
  id bigint,
  content text,
  summary text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  
  with vector_candidates as (
    select 
      lk.id,
      lk.content,
      lk.summary,
      lk.metadata,
      -- Calculate cosine similarity (1 - distance)
      -- Boost vector similarity to range 0.6-1.0 (above FTS baseline of 0.5)
      0.4 + (0.6 * (1 - (lk.embedding <=> query_embedding))) as similarity
    from legal_knowledge lk
    where (source_filter is null or (lk.metadata->>'source') = source_filter)
      and lk.embedding is not null
      and 1 - (lk.embedding <=> query_embedding) > match_threshold
    order by lk.embedding <=> query_embedding
    limit match_count
  ),
  
  keyword_candidates as (
    select 
      lk.id,
      lk.content,
      lk.summary,
      lk.metadata,
      -- FTS baseline score (lower than boosted vector scores)
      0.3::float as similarity
    from legal_knowledge lk
    where (source_filter is null or (lk.metadata->>'source') = source_filter)
      and length(text_search_query) > 0
      and to_tsvector('simple', lk.content) @@ to_tsquery('simple', text_search_query)
    limit match_count
  ),
  
  combined as (
    select * from vector_candidates
    union all
    select * from keyword_candidates
  ),
  
  deduplicated as (
    select distinct on (c.id)
      c.id,
      c.content,
      c.summary,
      c.metadata,
      c.similarity
    from combined c
    order by c.id, c.similarity desc
  )
  
  select * from deduplicated
  order by similarity desc
  limit match_count;

end;
$$;
