import sys
import os

# Add backend to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.services.llm_service import LLMService
    print("Successfully imported LLMService")

    print("Testing LLMService.get_config()...")
    config = LLMService.get_config()
    print("Success! Config retrieved:", config)
    
    # Also test fetch_models as that was likely affected too
    print("Testing LLMService._get_provider() instantiation...")
    provider = LLMService._get_provider()
    print(f"Success! Provider instantiated: {type(provider).__name__}")

except ImportError as e:
    print(f"FAIL: ImportError: {e}")
    sys.exit(1)
except AttributeError as e:
    print("FAIL: AttributeError caught:", e)
    sys.exit(1)
except Exception as e:
    print(f"FAIL: Unexpected error: {e}")
    # Print traceback
    import traceback
    traceback.print_exc()
    sys.exit(1)
