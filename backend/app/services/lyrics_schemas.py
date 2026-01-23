from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator

class FormattedLyricsSection(BaseModel):
    """
    Represents a single section of a song (e.g. Verse 1, Chorus).
    """
    section_type: str = Field(..., description="The type of section, e.g. 'Verse', 'Chorus', 'Bridge', 'Intro', 'Outro'.")
    section_index: Optional[int] = Field(None, description="The index of the section if numbered (e.g. 1 for Verse 1). None for unique sections like Intro.")
    content: str = Field(..., description="The actual lyrics content of the section, lines separated by newline.")

class LyricsEditOp(BaseModel):
    """
    A single operation to modify the song structure.
    """
    # Robust aliases for common LLM hallucinations
    op_type: Literal["UPDATE_SECTION", "INSERT_SECTION", "DELETE_SECTION", "APPEND_SECTION", "APPEND_CONTENT"] = Field(
        ..., 
        description="The type of operation to perform.",
        validation_alias="type" 
    )
    
    # Target identifiers
    target_section_type: str = Field(
        ..., 
        description="The type of the target section (e.g. 'Verse').",
        validation_alias="target_type"
    )
    target_section_index: Optional[int] = Field(
        None, 
        description="The index of the target section (e.g. 1). Use 1-based indexing.",
        validation_alias="target_index"
    )
    
    # New Content (for UPDATE, INSERT, APPEND)
    new_content: Optional[str] = Field(None, description="The new lyrics content. Required for UPDATE, INSERT, APPEND.")
    new_section_type: Optional[str] = Field(None, description="The type of the NEW section being inserted/appended (if different from target).")
    
    # Formatting/Placement details
    insert_position: Optional[Literal["BEFORE", "AFTER"]] = Field("AFTER", description="For INSERT_SECTION: whether to insert before or after the target.")

    # Allow extra fields to prevent validation error if LLM adds random stuff
    model_config = {
        "extra": "ignore",
        "populate_by_name": True
    }
    
    @field_validator('new_content', mode='before')
    @classmethod
    def allow_list_for_content(cls, v: Any) -> Optional[str]:
        if isinstance(v, list):
            return "\n".join(str(item) for item in v)
        return v

class LyricsResponse(BaseModel):
    """
    The structured response from the LLM containing the plan of edits.
    """
    thought_process: str = Field(..., description="Brief explaination of why these changes are being made.", validation_alias="thought")
    operations: List[LyricsEditOp] = Field(..., description="List of operations to apply to the song.")

    model_config = {
        "extra": "ignore",
        "populate_by_name": True
    }
