"""CI gate: every backend route needs a frontend caller (or exemption),
and every frontend call must resolve to a backend route.

Catches Training-Studio-class drift (UI buttons -> 404) at test time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))

from check_api_parity import check


def test_api_ui_parity():
    orphan_routes, dangling_calls = check()
    assert not orphan_routes, f"backend routes with no frontend caller: {orphan_routes}"
    assert not dangling_calls, f"frontend calls with no backend route: {dangling_calls}"
