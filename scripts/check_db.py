
import sys
import os
import logging

# Add parent directory to path for imports
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.unified_config import get_config
from supabase import create_client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPABASE_URL = get_config("SUPABASE_URL")
SUPABASE_KEY = get_config("SUPABASE_KEY")

def check_db():
    if not all([SUPABASE_URL, SUPABASE_KEY]):
        print("Missing Supabase credentials")
        return

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    try:
        # Count total articles
        result = supabase.table("legal_knowledge").select("id, created_at", count="exact").order("created_at", desc=True).limit(1).execute()
        print(f"Total documents: {result.count}")
        if result.data:
            print(f"Latest document created at: {result.data[0]['created_at']}")
        
        # Breakdown by source
        result = supabase.table("legal_knowledge").select("metadata").execute()
        sources = {}
        for row in result.data:
            meta = row.get('metadata', {})
            source = meta.get('source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
            
        print("\nBreakdown by source:")
        for source, count in sources.items():
            print(f"- {source}: {count} documents")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_db()
