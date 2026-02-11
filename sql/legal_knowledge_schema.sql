-- ============================================
-- FRIDAY Legal Brain - Database Schema
-- Run this in Supabase SQL Editor
-- ============================================

-- Enable pgvector if not already enabled
create extension if not exists vector;

-- Create legal_knowledge table
create table if not exists legal_knowledge (
  id bigserial primary key,
  content text not null,                    -- Full extracted text
  summary text,                             -- AI-generated summary
  metadata jsonb default '{}'::jsonb,       -- { "source": "PC200", "topic": "Wages", "effective_date": "2024-01-01" }
  content_hash text unique,                 -- MD5 for deduplication
  embedding vector(384),                    -- Same dimension as paraphrase-multilingual-MiniLM-L12-v2
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Indexes for fast querying
create index if not exists idx_legal_knowledge_source 
  on legal_knowledge ((metadata->>'source'));
create index if not exists idx_legal_knowledge_effective_date 
  on legal_knowledge ((metadata->>'effective_date'));
create index if not exists idx_legal_knowledge_hash 
  on legal_knowledge (content_hash);
create index if not exists idx_legal_knowledge_embedding 
  on legal_knowledge using ivfflat (embedding vector_cosine_ops) with (lists = 50);

-- Indexes for article-level lookup
create index if not exists idx_legal_knowledge_law_name
  on legal_knowledge ((metadata->>'law_name'));
create index if not exists idx_legal_knowledge_article
  on legal_knowledge ((metadata->>'article_number'));

-- Full-text search index
create index if not exists idx_legal_knowledge_fts 
  on legal_knowledge using gin (to_tsvector('simple', content));

-- RPC function for hybrid search with recency boost
create or replace function match_legal_documents(
  query_embedding vector(384),
  text_search_query text,      -- Pipe-delimited: 'term1 | term2'
  match_threshold float default 0.15,
  match_count int default 10,
  source_filter text default null  -- Filter by source like 'PC200'
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
      -- Recency boost: newer documents score higher
      (1 - (lk.embedding <=> query_embedding)) * 
        (1 + 0.1 * (1 - extract(epoch from (now() - lk.created_at)) / 31536000)) as similarity
    from legal_knowledge lk
    where (source_filter is null or (lk.metadata->>'source') = source_filter)
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
      0.5::float as similarity
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

-- Grant permissions
grant select, insert, update on legal_knowledge to anon, authenticated;
grant usage, select on sequence legal_knowledge_id_seq to anon, authenticated;
grant execute on function match_legal_documents to anon, authenticated;
