import logging
from typing import List, Dict, Any
from .lyrics_utils import LyricsDOM
from .lyrics_schemas import LyricsEditOp

logger = logging.getLogger(__name__)

class StructuredLyricsEngine:
    """
    Engine for applying structured edits to lyrics using LyricsDOM.
    Replaces the old fuzzy-logic "SotaLyricsPatcher".
    """
    
    def apply_edits(self, original_text: str, operations: List[LyricsEditOp]) -> str:
        """
        Applies a list of Pydantic-model operations to the text.
        Returns the new lyrics string.
        """
        if not operations:
            return original_text
            
        try:
            dom = LyricsDOM(original_text)
            
            # Log structure before
            logger.info(f"Structure before edit: {dom.get_structure_map()}")
            
            # Sanitize content in ops
            for op in operations:
                if op.new_content:
                    clean = op.new_content.strip()
                    # Remove surrounding quotes if present
                    if clean.startswith('"') and clean.endswith('"'):
                        clean = clean[1:-1].strip()

                    # INTELLIGENT SANITIZATION:
                    # The AI often includes the header "[Verse 1]" in the content.
                    # We must remove this specific line to avoid duplication.
                    
                    lines = clean.split('\n')
                    if lines and lines[0].strip().startswith('[') and lines[0].strip().endswith(']'):
                        # Detected a header line at the start. Remove it.
                        logger.info(f"Sanitizer: Removed duplicated header line '{lines[0]}'")
                        lines = lines[1:]
                        clean = "\n".join(lines).strip()
                    
                    # Fallback for partial stripping (e.g. if AI did "Verse 1]")
                    # Only apply if it looks strictly like a malformed tag at the very start
                    # Example: "Verse 1]\nLine 1"
                    if ']' in clean.split('\n')[0] and '[' not in clean.split('\n')[0]:
                         first_line = clean.split('\n')[0]
                         # Heuristic: if it's short (< 20 chars) and has a closing bracket, it's likely a broken header
                         if len(first_line) < 20: 
                             logger.info(f"Sanitizer: Removed likely broken header line '{first_line}'")
                             lines = clean.split('\n')[1:]
                             clean = "\n".join(lines).strip()

                    op.new_content = clean

            # Apply ops
            dom.apply_ops(operations)
            
            # Log structure after
            logger.info(f"Structure after edit: {dom.get_structure_map()}")
            
            return dom.render()
            
        except Exception as e:
            logger.error(f"Structured Engine Failed: {e}")
            # Fallback: validation should catch this, but if runtime error, return original
            return original_text
