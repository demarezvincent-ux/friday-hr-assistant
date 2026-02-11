import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

class UnifiedConfig:
    """
    Unified configuration loader that handles:
    1. Environment variables (Local/CI/Docker)
    2. Streamlit secrets (.streamlit/secrets.toml)
    """
    
    _secrets = {}
    _loaded = False
    
    @classmethod
    def load(cls):
        """Loads secrets from .streamlit/secrets.toml if it exists."""
        if cls._loaded:
            return
            
        secrets_path = os.path.join(os.getcwd(), '.streamlit', 'secrets.toml')
        if os.path.exists(secrets_path):
            try:
                import tomllib
                with open(secrets_path, "rb") as f:
                    cls._secrets = tomllib.load(f)
                logger.info(f"✓ Loaded config from {secrets_path}")
            except Exception as e:
                logger.warning(f"⚠️ Could not load {secrets_path}: {e}")
        
        cls._loaded = True

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.
        Priority: 
        1. Environment variable
        2. Streamlit secrets.toml
        3. Default value
        """
        # Ensure we've tried to load secrets
        if not cls._loaded:
            cls.load()
            
        # 1. Try Environment Variable (Production/Docker preference)
        env_val = os.environ.get(key)
        if env_val:
            return env_val
            
        # 2. Try loaded secrets (Local development preference)
        # Note: Streamlit secrets are usually flat, but can be nested if using sections
        return cls._secrets.get(key, default)

# Global helper for quick access
def get_config(key: str, default: Any = None) -> Any:
    return UnifiedConfig.get(key, default)
