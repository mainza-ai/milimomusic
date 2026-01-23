import re
import difflib
from typing import Dict, List, Optional, Tuple, Any
from .lyrics_schemas import FormattedLyricsSection, LyricsEditOp

class LyricsParser:
    @staticmethod
    def parse_to_dom(lyrics_text: str) -> List[FormattedLyricsSection]:
        """
        Parses lyrics text into a list of FormattedLyricsSection objects.
        Robustly handles missing brackets, numbering, etc.
        """
        if not lyrics_text:
            return []

        # Strategy: Split by lines, look for [Header] patterns.
        # Everything until the next header is content.
        
        sections: List[FormattedLyricsSection] = []
        lines = lyrics_text.split('\n')
        
        current_header_type = "Verse" # Default if no header starts
        current_header_index = 1
        current_content_lines = []
        
        # Regex for headers like [Verse 1], [Chorus], [Bridge], [Verse 2]
        header_pattern = re.compile(r"^\[(.*?)(?:\s+(\d+))?\]$")
        
        # Pre-scan: if the first line IS a header, we start correctly.
        # If not, we accumulate into a "Start" or "Verse 1" implicit section.
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Clean Artifacts
            if line.startswith("'''") or line.endswith("'''"):
                line = line.replace("'''", "").strip()
            
            # Regex for "Existing Lyrics" artifacts (typo tolerant)
            # Matches [Existing Lyrics], [Existling Lyrics], Existing Lyrics: etc.
            if re.match(r"^\[?Exist.*?Lyrics.*?\]?$", line, re.IGNORECASE):
                continue
                
            match = header_pattern.match(line)
            
            if match:
                # Flush previous section if it has content
                if current_content_lines or (i > 0): # Don't flush empty start unless it was an explicit section
                    if sections or current_content_lines: # Only add if we have something or it's not the very start empty
                         # If it's the very first implicit section and empty, skip it?
                         # Actually, let's just save whatever we have.
                         if not sections and not current_content_lines:
                             pass # Start of file, first header found immediately
                         else:
                            sections.append(FormattedLyricsSection(
                                section_type=current_header_type,
                                section_index=current_header_index,
                                content="\n".join(current_content_lines).strip()
                            ))
                
                # Start new section
                raw_type = match.group(1).strip()
                raw_index = match.group(2)
                
                current_header_type = raw_type
                current_header_index = int(raw_index) if raw_index else None
                current_content_lines = []
            else:
                current_content_lines.append(line)
                
        # Flush final section
        if current_content_lines or sections:
            sections.append(FormattedLyricsSection(
                section_type=current_header_type,
                section_index=current_header_index,
                content="\n".join(current_content_lines).strip()
            ))
            
        # Post-processing: If we have multiple "Verse" without indexes, we might want to auto-number them?
        # But for now, trust the parser or the user's text.
        
        return sections

    @staticmethod
    def to_string(sections: List[FormattedLyricsSection]) -> str:
        output = []
        for s in sections:
            header = f"[{s.section_type}]"
            if s.section_index is not None:
                header = f"[{s.section_type} {s.section_index}]"
            
            output.append(header)
            if s.content:
                output.append(s.content)
            output.append("") # Blank line after section
            
        return "\n".join(output).strip()


class LyricsDOM:
    def __init__(self, lyrics_text: str):
        self.sections = LyricsParser.parse_to_dom(lyrics_text)

    def render(self) -> str:
        return LyricsParser.to_string(self.sections)
    
    def get_structure_map(self) -> str:
        """Returns a string description of structure like 'Verse 1, Chorus, Verse 2'."""
        parts = []
        for s in self.sections:
            name = s.section_type
            if s.section_index:
                name += f" {s.section_index}"
            parts.append(name)
        return ", ".join(parts)

    def apply_ops(self, ops: List[LyricsEditOp]):
        """
        Applies a list of structured operations.
        Operations are applied sequentially.
        """
        for op in ops:
            try:
                self._apply_single_op(op)
            except Exception as e:
                print(f"Failed to apply op {op}: {e}")
                # We continue to try others? Or stop? 
                # Better to continue for robustness.

    def _find_section_index(self, s_type: str, s_index: Optional[int]) -> int:
        """
        Finds the list index of a section matching type and index.
        Robust strategy:
        1. Try exact match on section_index (e.g. [Verse 2] matches index=2)
        2. Fallback: Try N-th occurrence of type (e.g. index=2 matches 2nd Verse found)
        """
        target_type = s_type.lower().strip()
        desired_idx = s_index if s_index is not None else 1
        
        # Strategy 1: Exact Property Match
        for i, sec in enumerate(self.sections):
            curr_type = sec.section_type.lower().strip()
            if curr_type == target_type:
                # Match if explicit index matches
                if sec.section_index == s_index:
                    return i
                # Match if we want index 1 (default) and section has no index
                if (desired_idx == 1) and sec.section_index is None:
                    return i

        # Strategy 2: N-th Occurrence Match (Fallback)
        # If LLM counts "Verse 1" as "Verse 2" because it's the 2nd block, this won't help.
        # But if LLM counts "Verse 1" as "1st Verse" (index 1) and our parser said index=None, this helps.
        # Or if LLM says "Verse 2" and parser said index=None but it's the 2nd one.
        
        count = 0
        for i, sec in enumerate(self.sections):
            curr_type = sec.section_type.lower().strip()
            if curr_type == target_type:
                count += 1
                if count == desired_idx:
                    return i
                    
        return -1

    def _apply_single_op(self, op: LyricsEditOp):
        if op.op_type == "UPDATE_SECTION":
            idx = self._find_section_index(op.target_section_type, op.target_section_index)
            if idx != -1:
                self.sections[idx].content = op.new_content
        
        elif op.op_type == "DELETE_SECTION":
            idx = self._find_section_index(op.target_section_type, op.target_section_index)
            if idx != -1:
                del self.sections[idx]
                
        elif op.op_type == "INSERT_SECTION" or op.op_type == "APPEND_SECTION":
            # For APPEND, we can treat as INSERT AFTER LAST if target not found?
            # Or INSERT relative to target.
            
            new_sec = FormattedLyricsSection(
                section_type=op.new_section_type or op.target_section_type, # Default to target type if not spec
                section_index=None, # We'll re-index later or let user specify? Let's default None.
                content=op.new_content
            )
            
            # If APPEND to end
            if op.op_type == "APPEND_SECTION":
                self.sections.append(new_sec)
                return

            # If INSERT relative
            idx = self._find_section_index(op.target_section_type, op.target_section_index)
            if idx != -1:
                if op.insert_position == "BEFORE":
                    self.sections.insert(idx, new_sec)
                else:
                    self.sections.insert(idx + 1, new_sec)
            else:
                # Target not found? Append.
                self.sections.append(new_sec)

        elif op.op_type == "APPEND_CONTENT":
            idx = self._find_section_index(op.target_section_type, op.target_section_index)
            if idx != -1:
                original = self.sections[idx].content.strip()
                to_add = op.new_content.strip() if op.new_content else ""
                
                if original:
                    self.sections[idx].content = f"{original}\n{to_add}"
                else:
                    self.sections[idx].content = to_add

        # Re-Index logic could go here if we wanted to auto-renumber "Verse 1, Verse 2" etc.
        # For now, let's keep it simple.

