import sys
import os

# Adjust path to find backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.lyrics_utils import LyricsDOM
from app.services.lyrics_schemas import LyricsEditOp

def test_parser():
    print("MATCH 1: Parsing")
    raw = """[Verse 1]
Hello world
This is a test

[Chorus]
Singing allow
Loudly"""
    dom = LyricsDOM(raw)
    structure = dom.get_structure_map()
    print(f"Structure: {structure}")
    assert "Verse 1" in structure
    assert "Chorus" in structure
    print("PASS\n")

def test_update():
    print("MATCH 2: Update Section")
    raw = "[Verse 1]\nOld Line"
    dom = LyricsDOM(raw)
    
    op = LyricsEditOp(
        op_type="UPDATE_SECTION",
        target_section_type="Verse",
        target_section_index=1,
        new_content="New Line Updated"
    )
    dom.apply_ops([op])
    result = dom.render()
    print(f"Result:\n{result}")
    assert "New Line Updated" in result
    assert "Old Line" not in result
    print("PASS\n")

def test_insert_middle():
    print("MATCH 3: Insert Section (Bridge after Verse 1)")
    raw = """[Verse 1]
A
[Chorus]
B"""
    dom = LyricsDOM(raw)
    
    op = LyricsEditOp(
        op_type="INSERT_SECTION",
        target_section_type="Verse",
        target_section_index=1,
        insert_position="AFTER",
        new_section_type="Bridge",
        new_content="I am a bridge"
    )
    dom.apply_ops([op])
    result = dom.render()
    print(f"Result:\n{result}")
    
    # Check order: Verse 1 -> Bridge -> Chorus
    lines = result.split('\n')
    # Filter empty
    lines = [l for l in lines if l.strip()]
    
    # Very rough check
    idx_v1 = -1
    idx_bridge = -1
    idx_chorus = -1
    
    for i, line in enumerate(lines):
        if "[Verse 1]" in line: idx_v1 = i
        if "[Bridge]" in line: idx_bridge = i
        if "[Chorus]" in line: idx_chorus = i
        
    assert idx_v1 < idx_bridge < idx_chorus
    print("PASS\n")

def test_delete():
    print("MATCH 4: Delete Section")
    raw = "[Intro]\nA\n[Verse 1]\nB"
    dom = LyricsDOM(raw)
    
    op = LyricsEditOp(
        op_type="DELETE_SECTION",
        target_section_type="Intro"
    )
    dom.apply_ops([op])
    result = dom.render()
    print(f"Result:\n{result}")
    assert "[Intro]" not in result
    assert "[Verse 1]" in result
    print("PASS\n")

if __name__ == "__main__":
    test_parser()
    test_update()
    test_insert_middle()
    test_delete()
