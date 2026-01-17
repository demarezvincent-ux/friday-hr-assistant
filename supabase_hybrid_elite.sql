create or replace function match_documents_hybrid_elite(
  query_embedding vector(384),
  text_search_query text,      -- Expects pipe-delimited: 'term1 | term2 | term3'
  match_threshold float,
  match_count int,
  company_id_filter text
)
returns table (
  id bigint,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  
  -- Vector similarity candidates
  with vector_candidates as (
    select 
      dc.id,
      dc.content,
      dc.metadata,
      1 - (dc.embedding <=> query_embedding) as similarity
    from document_chunks dc
    where (dc.metadata->>'company_id') = company_id_filter
      and (dc.metadata->>'is_active')::boolean = true
      and 1 - (dc.embedding <=> query_embedding) > match_threshold
    order by dc.embedding <=> query_embedding
    limit match_count
  ),
  
  -- Full-text search candidates (using to_tsquery for pipe syntax)
  keyword_candidates as (
    select 
      dc.id,
      dc.content,
      dc.metadata,
      0.5::float as similarity  -- Fixed score for keyword matches
    from document_chunks dc
    where (dc.metadata->>'company_id') = company_id_filter
      and (dc.metadata->>'is_active')::boolean = true
      and to_tsvector('simple', dc.content) @@ to_tsquery('simple', text_search_query)
    limit match_count
  ),
  
  -- Combine and deduplicate by ID, keeping highest similarity
  combined as (
    select * from vector_candidates
    union all
    select * from keyword_candidates
  ),
  
  deduplicated as (
    select distinct on (c.id)
      c.id,
      c.content,
      c.metadata,
      c.similarity
    from combined c
    order by c.id, c.similarity desc
  )
  
  -- Return sorted by similarity
  select * from deduplicated
  order by similarity desc
  limit match_count;

end;
$$;
