"""
Multi-Agent Lyrics Engine using pydantic-graph.

This module implements a graph-based workflow for lyrics editing with:
- CoordinatorNode: Routes between CREATION and EDIT modes
- LyricistNode: Creative agent for lyrics generation
- StructureGuardNode: QA agent for validation with automatic retry
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Literal, Any, Union

from pydantic_graph import BaseNode, Graph, GraphRunContext, End

from .lyrics_schemas import LyricsResponse, LyricsEditOp
from .lyrics_engine import StructuredLyricsEngine
from .lyrics_utils import LyricsDOM
from .config_manager import ConfigManager

logger = logging.getLogger(__name__)


def debug_log(message: str, data: Any = None):
    """Write detailed debug info to ai_debug.log"""
    try:
        with open("ai_debug.log", "a") as f:
            timestamp = datetime.now().strftime("%H:%M:%S")
            f.write(f"[{timestamp}] {message}\n")
            if data is not None:
                f.write(f"{data}\n")
    except Exception as e:
        logger.warning(f"Debug log failed: {e}")


def sanitize_lyrics(raw_lyrics: str) -> str:
    """
    Post-process lyrics to ensure they comply with the expected format.
    
    Fixes:
    1. Convert "Intro:" or "Verse 1:" to "[Intro]" or "[Verse 1]"
    2. Remove instrumental directions like "(Soft piano melody)"
    3. Remove stage directions like "(repeat chorus)"
    """
    lines = raw_lyrics.split('\n')
    cleaned_lines = []
    
    # Pattern for wrong header format: "Intro:" or "Verse 1:" etc
    wrong_header_pattern = re.compile(r'^(Intro|Verse|Chorus|Bridge|Outro|Pre-Chorus|Hook)(\s*\d*):\s*$', re.IGNORECASE)
    
    # Pattern for instrumental/stage directions: anything in parentheses
    instrumental_pattern = re.compile(r'^\s*\(.*?\)\s*$')
    
    # Pattern for inline instrumental directions
    inline_instrumental_pattern = re.compile(r'\([^)]*(?:melody|solo|instrumental|break|repeat|guitar|piano|drums|strings|music)[^)]*\)', re.IGNORECASE)
    
    for line in lines:
        # Fix wrong header format
        match = wrong_header_pattern.match(line.strip())
        if match:
            section_type = match.group(1).title()
            section_num = match.group(2).strip()
            if section_num:
                cleaned_lines.append(f'[{section_type} {section_num}]')
            else:
                cleaned_lines.append(f'[{section_type}]')
            continue
        
        # Skip lines that are purely instrumental directions
        if instrumental_pattern.match(line.strip()):
            continue
        
        # Remove inline instrumental directions
        cleaned_line = inline_instrumental_pattern.sub('', line)
        
        # Clean up any double spaces created
        cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
        
        # Skip empty lines that resulted from cleaning (but keep intentional blank lines)
        if line.strip() and not cleaned_line:
            continue
            
        cleaned_lines.append(cleaned_line if cleaned_line else line)
    
    return '\n'.join(cleaned_lines)

# ============================================================================
# State & Dependencies
# ============================================================================

@dataclass
class SongState:
    """Persistent state passed between graph nodes."""
    # Context from user request
    topic: str
    style_tags: str
    current_lyrics: str
    original_request: str
    
    # Computed
    structure_map: str = ""
    
    # Execution tracking
    attempts: int = 0
    max_attempts: int = 3
    history: List[str] = field(default_factory=list)
    
    # Feedback loop
    error_feedback: Optional[str] = None
    
    # Output
    final_lyrics: Optional[str] = None
    final_message: Optional[str] = None


@dataclass  
class GraphDeps:
    """Dependencies injected into graph nodes."""
    config_manager: ConfigManager
    provider: Any  # LLMProvider instance
    model_name: str


# ============================================================================
# Prompts
# ============================================================================

LYRICIST_CREATION_PROMPT = """
ROLE: You are an award-winning professional songwriter.
GOAL: Write a COMPLETE song based on the user's request and seed content.

SONG CONCEPT: {topic}
STYLE/GENRE: {style}

USER REQUEST: {request}
SEED CONTENT (if any): {seed}

STRICT FORMAT REQUIREMENTS:
1. SEED CONTENT RULE: You MUST start the song with the EXACT seed content provided.
   - If the seed is a lyric line (e.g., "I'm an alien"), put it INSIDE the first section (e.g., under [Verse 1]).
   - Do NOT wrap seed content in square brackets [ ].
   - Do NOT make the seed content a section header.
2. Section headers MUST use square brackets: [Intro], [Verse 1], [Chorus], [Bridge], [Outro]
   - CORRECT: [Intro]
   - WRONG: Intro:
   - WRONG: **Intro**
3. NO instrumental directions or stage notations:
   - FORBIDDEN: (Soft piano melody)
   - FORBIDDEN: (Guitar solo)
   - FORBIDDEN: (Instrumental break)
   - FORBIDDEN: (Repeat chorus)
4. Output ONLY singable lyrics - every line must be text that a vocalist can sing
5. Use vivid imagery and metaphor - show don't tell
6. Ensure consistent rhythm and meter across verses

EXAMPLE FORMAT:
[Intro]
Opening lyrics here that set the mood

[Verse 1]
First verse lyrics with imagery
Second line continuing the story

[Chorus]
Catchy memorable chorus lyrics
That capture the song's essence

[Bridge]
A musical and lyrical shift

[Outro]
Closing lyrics that wrap up the song

OUTPUT: Provide ONLY the formatted lyrics with section headers in square brackets. No explanations, no instrumental notations.

{error_feedback}
"""

LYRICIST_EDIT_PROMPT = """
ROLE: You are an award-winning professional songwriter.
GOAL: Modify the existing lyrics based on the user's specific request.

SONG CONCEPT: {topic}
STYLE/GENRE: {style}
CURRENT STRUCTURE: {structure_map}

CURRENT LYRICS:
'''
{current_lyrics}
'''

USER REQUEST: "{request}"

OPERATION SELECTION GUIDE:
- APPEND_CONTENT: Add NEW lines to END of an existing section (e.g., "add two lines to the outro")
  → new_content = ONLY the new lines to add, NOT the original section content
- UPDATE_SECTION: Replace an entire section with completely new content
- INSERT_SECTION: Create a BRAND NEW section type (e.g., "add a bridge between verse 2 and chorus")
- DELETE_SECTION: Remove an entire section

CRITICAL RULES:
1. If user says "add lines to [section]" → Use APPEND_CONTENT with ONLY the new lines
2. For APPEND_CONTENT: new_content should contain ONLY the lines being added, NOT a copy of the original section
3. NEVER duplicate existing lyrics in new_content

REQUIRED JSON FORMAT:
{{
  "thought_process": "Brief explanation of your plan...",
  "operations": [
    {{
      "op_type": "UPDATE_SECTION" | "INSERT_SECTION" | "DELETE_SECTION" | "APPEND_CONTENT",
      "target_section_type": "Intro" | "Verse" | "Chorus" | "Bridge" | "Outro" | etc.,
      "target_section_index": 1,  // Use 1-based indexing (e.g., Verse 1 = index 1, Intro = 1)
      "new_content": "ONLY the new lyrics as a single string with \\n separators. DO NOT use a list of strings.",
      "new_section_type": "Bridge",  // Only for INSERT_SECTION
      "insert_position": "BEFORE" | "AFTER"  // Only for INSERT_SECTION
    }}
  ]
}}

EXAMPLE - User says "add two lines to the outro":
WRONG: new_content = ["New line 1", "New line 2"]
WRONG: new_content = "Original line 1\\nOriginal line 2\\nNew line 1\\nNew line 2"
CORRECT: new_content = "New line 1\\nNew line 2"

RULES:
1. MANDATORY FIELDS: 'op_type', 'target_section_type', 'target_section_index' are ALWAYS required.
2. CONTENT FORMAT: 'new_content' MUST be a single string. NOT a list.
3. NO HEADERS: NEVER include "[Verse 1]" inside new_content.
4. NO STAGE DIRECTIONS: No "(guitar solo)", etc.
5. MINIMAL changes: Only touch sections user explicitly asked to change

{error_feedback}
"""

STRUCTURE_GUARD_PROMPT = """
ROLE: You are a QA engineer validating lyrics operations.
GOAL: Convert the raw lyrics draft into properly structured JSON operations.

RULES:
1. No duplicate headers - strip "[Verse 1]" from content if present at start
2. Bleeding check - reject if a single content block contains multiple sections
3. Verify all required fields are present
4. Output strictly the LyricsResponse format

Input to validate:
{raw_draft}

If the input is already valid JSON, clean it up and output.
If it's raw lyrics text, convert to appropriate operations.
"""


# ============================================================================
# Graph Nodes
# ============================================================================

@dataclass
class LyricistNode(BaseNode[SongState, GraphDeps, LyricsResponse]):
    """Creative agent that generates/edits lyrics."""
    user_request: str
    mode: Literal["CREATION", "EDIT"] = "EDIT"
    
    async def run(
        self, 
        ctx: GraphRunContext[SongState, GraphDeps]
    ) -> Union["StructureGuardNode", End[LyricsResponse]]:
        state = ctx.state
        deps = ctx.deps
        
        # Check retry limit
        if state.attempts >= state.max_attempts:
            state.history.append(f"Lyricist: Max retries ({state.max_attempts}) exceeded")
            logger.warning(f"Max retries exceeded for lyrics chat")
            return End(LyricsResponse(
                thought_process="Failed after maximum retries",
                operations=[]
            ))
        
        state.attempts += 1
        state.history.append(f"Lyricist: Attempt {state.attempts}, mode={self.mode}")
        
        error_section = f"PREVIOUS ERROR TO FIX: {state.error_feedback}" if state.error_feedback else ""
        
        if self.mode == "CREATION":
            # Full generation mode - bypass structured ops
            prompt = LYRICIST_CREATION_PROMPT.format(
                topic=state.topic,
                style=state.style_tags,
                request=self.user_request,
                seed=state.current_lyrics,
                error_feedback=error_section
            )
            
            try:
                raw_lyrics = deps.provider.generate_text(prompt, deps.model_name)
                # Sanitize output to fix format issues
                cleaned_lyrics = sanitize_lyrics(raw_lyrics)
                state.final_lyrics = cleaned_lyrics
                state.final_message = "I've created a full song based on your request."
                
                return End(LyricsResponse(
                    thought_process="Full song generation completed",
                    operations=[]
                ))
            except Exception as e:
                logger.error(f"Lyricist CREATION failed: {e}")
                state.error_feedback = str(e)
                return LyricistNode(user_request=self.user_request, mode="CREATION")
        
        else:  # EDIT mode
            prompt = LYRICIST_EDIT_PROMPT.format(
                topic=state.topic,
                style=state.style_tags,
                structure_map=state.structure_map,
                current_lyrics=state.current_lyrics,
                request=self.user_request,
                error_feedback=error_section
            )
            
            try:
                # Get JSON response
                result: LyricsResponse = deps.provider.generate_structured(
                    prompt, 
                    deps.model_name, 
                    LyricsResponse,
                    options={"temperature": 0.4}
                )
                
                # Detailed debug logging
                debug_log(f"=== AI RESPONSE (Attempt {state.attempts}) ===")
                debug_log(f"Thought Process: {result.thought_process}")
                for i, op in enumerate(result.operations):
                    debug_log(f"Operation {i+1}: {op.op_type} on {op.target_section_type} (index={op.target_section_index})")
                    if op.new_content:
                        debug_log(f"  new_content ({len(op.new_content)} chars):", op.new_content[:500] + ("..." if len(op.new_content) > 500 else ""))
                    if op.new_section_type:
                        debug_log(f"  new_section_type: {op.new_section_type}")
                    if op.insert_position:
                        debug_log(f"  insert_position: {op.insert_position}")
                debug_log("=== END AI RESPONSE ===\n")
                
                # Pass to guard for validation
                return StructureGuardNode(parsed_response=result)
                
            except Exception as e:
                logger.error(f"Lyricist EDIT failed: {e}")
                state.error_feedback = f"JSON parsing failed: {str(e)}. Please output valid JSON only."
                return LyricistNode(user_request=self.user_request, mode="EDIT")


@dataclass
class StructureGuardNode(BaseNode[SongState, GraphDeps, LyricsResponse]):
    """QA agent that validates and applies structured operations."""
    parsed_response: LyricsResponse
    
    async def run(
        self,
        ctx: GraphRunContext[SongState, GraphDeps]
    ) -> Union[LyricistNode, End[LyricsResponse]]:
        state = ctx.state
        
        state.history.append(f"Guard: Validating {len(self.parsed_response.operations)} operations")
        
        try:
            # Deep validation
            validation_error = self._deep_validate(self.parsed_response)
            if validation_error:
                raise ValueError(validation_error)
            
            # Apply operations
            engine = StructuredLyricsEngine()
            new_lyrics = engine.apply_edits(state.current_lyrics, self.parsed_response.operations)
            
            state.final_lyrics = new_lyrics
            state.final_message = self.parsed_response.thought_process
            state.history.append("Guard: Validation passed, operations applied")
            
            return End(self.parsed_response)
            
        except Exception as e:
            logger.warning(f"Guard validation failed: {e}")
            state.error_feedback = f"Validation failed: {str(e)}. Please fix and retry."
            
            # Return to lyricist for retry
            return LyricistNode(
                user_request=state.original_request,
                mode="EDIT"
            )
    
    def _deep_validate(self, data: LyricsResponse) -> Optional[str]:
        """Additional Python-side validation beyond Pydantic."""
        for op in data.operations:
            # Check for headers inside content
            if op.new_content:
                lines = op.new_content.strip().split('\n')
                if lines and lines[0].strip().startswith('[') and lines[0].strip().endswith(']'):
                    return f"Content contains section header '{lines[0]}' which should be removed"
                    
            # Check for bleeding (multiple sections in one content)
            if op.new_content and op.new_content.count('[') > 1:
                return "Content contains multiple section headers - this suggests bleeding"
                
        return None


@dataclass  
class CoordinatorNode(BaseNode[SongState, GraphDeps, LyricsResponse]):
    """Router node that decides between CREATION and EDIT modes."""
    user_request: str
    
    async def run(
        self,
        ctx: GraphRunContext[SongState, GraphDeps]
    ) -> LyricistNode:
        state = ctx.state
        
        # Compute structure map
        dom = LyricsDOM(state.current_lyrics)
        state.structure_map = dom.get_structure_map()
        state.original_request = self.user_request
        
        # Short lyrics bypass - use CREATION mode
        is_short = len(state.current_lyrics) < 150 or state.current_lyrics.count('\n') < 3
        
        if is_short:
            state.history.append("Coordinator: Short/empty lyrics detected -> CREATION mode")
            logger.info("Short lyrics bypass activated")
            return LyricistNode(
                user_request=self.user_request,
                mode="CREATION"
            )
        
        state.history.append(f"Coordinator: Edit mode, structure={state.structure_map}")
        return LyricistNode(
            user_request=self.user_request,
            mode="EDIT"
        )


# ============================================================================
# Graph Definition
# ============================================================================

lyrics_graph = Graph(
    nodes=[CoordinatorNode, LyricistNode, StructureGuardNode],
    state_type=SongState,
)


# ============================================================================
# Public API
# ============================================================================

class MaxRetriesExceededError(Exception):
    """Raised when the graph fails after maximum retry attempts."""
    pass


async def run_lyrics_graph(
    current_lyrics: str,
    user_message: str,
    topic: Optional[str],
    tags: Optional[str],
    provider: Any,
    model_name: str
) -> dict:
    """
    Run the multi-agent lyrics graph.
    
    Returns:
        dict with 'message' and 'lyrics' keys
    
    Raises:
        MaxRetriesExceededError: If graph fails after max retries
    """
    state = SongState(
        topic=topic or "",
        style_tags=tags or "",
        current_lyrics=current_lyrics,
        original_request=user_message,
    )
    
    deps = GraphDeps(
        config_manager=ConfigManager(),
        provider=provider,
        model_name=model_name,
    )
    
    try:
        result = await lyrics_graph.run(
            CoordinatorNode(user_request=user_message),
            state=state,
            deps=deps,
        )
        
        # Check if we have final output
        if state.final_lyrics is not None:
            return {
                "message": state.final_message or "I've updated the lyrics.",
                "lyrics": state.final_lyrics
            }
        
        # If we got here with empty operations, max retries was hit
        if not result.output.operations:
            logger.error(f"Graph completed with no operations. History: {state.history}")
            raise MaxRetriesExceededError(
                "The AI was unable to process your request after multiple attempts. "
                "Please try rephrasing your request or making it more specific."
            )
        
        # This shouldn't happen, but fallback
        return {
            "message": result.output.thought_process,
            "lyrics": current_lyrics
        }
        
    except MaxRetriesExceededError:
        raise
    except Exception as e:
        logger.error(f"Lyrics graph failed: {e}")
        raise MaxRetriesExceededError(
            f"An unexpected error occurred while processing your request: {str(e)}"
        )
