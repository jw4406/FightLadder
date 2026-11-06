"""Phase-1 parity gate for the --combo_capacity action-space padding.

The change lets a MULTI-protagonist spar/ippo run share one homogeneous action space by padding the
special-move combo tail to a fixed width (instead of sizing it to the ego char's own combo count).
It MUST be a no-op for every existing run:

  * combo_capacity=None            -> legacy per-ego sizing (byte-identical action space + transform).
  * combo_capacity=len(ego combos) -> identical to None (padding to the ego's own count changes nothing).
  * existing rosters               -> auto-capacity == ego's own count (opponents never exceed ego here),
                                      so GuVeCh / GuCh / ... are unaffected even with --combo_capacity auto.

And it must correctly ENABLE the new case:

  * {Guile,ChunLi} x {Ryu,Guile,ChunLi} -> auto-capacity 6 -> every env action space is 63+6 = 69-wide,
    so the two ego chars (Guile 66, ChunLi 65) and the Ryu adversary (69) all share one space.

Run the light half (no emulator) anywhere:   python -c "import test_combo_capacity as t; t.test_capacity_math()"
Run the full parity half on a COMPUTE node:   python test_combo_capacity.py   (needs stable-retro + the ROM)
"""
from common.const import build_sf_combos


def _macros(char):
    return len(build_sf_combos(8, char))


def _resolve_capacity(cap_req, player, opponents, nsf=8):
    """Mirror of ippo.main()'s capacity resolution, isolated for testing."""
    if str(cap_req or "0").lower() in ("none", "0", ""):
        return None
    roster = list(dict.fromkeys(list(player) + list(opponents)))
    max_macros = max(len(build_sf_combos(nsf, c)) for c in roster)
    cap = max_macros if str(cap_req).lower() == "auto" else int(cap_req)
    assert cap >= max_macros, f"capacity {cap} < max macros {max_macros} over {roster}"
    assert cap <= 7, f"capacity {cap} > 7 (3-bit combo field)"
    return cap


def test_capacity_math():
    # Per-character combo counts (const.MOVESETS), the basis for action_dim = 63 + n_combo.
    assert _macros("Guile") == 3, _macros("Guile")
    assert _macros("ChunLi") == 2, _macros("ChunLi")
    assert _macros("Ryu") == 6, _macros("Ryu")
    assert _macros("Vega") == 2, _macros("Vega")

    # Legacy: no capacity requested -> None (every existing run path unchanged).
    assert _resolve_capacity(None, ["Guile"], ["Vega", "ChunLi"]) is None
    assert _resolve_capacity("0", ["ChunLi"], ["Guile", "ChunLi"]) is None

    # Backward-compat under --combo_capacity auto: existing rosters resolve to the EGO's own count,
    # so the padded width equals the legacy width (no behavioural change).
    #   Guile(3) vs {Vega(2),ChunLi(2)} -> 3  == Guile's own count  -> width stays 66.
    assert _resolve_capacity("auto", ["Guile"], ["Vega", "ChunLi"]) == 3
    #   NOTE: ChunLi(2) vs {Guile(3),ChunLi(2)} -> 3, which is LARGER than ChunLi's own 2. 'auto' would
    #   widen Ch_vs_GuCh from 65 to 66 (and let the Guile adversary use its 3rd combo). That is why
    #   capacity is OPT-IN per run: existing live runs pass no flag -> None -> stay 65.
    assert _resolve_capacity("auto", ["ChunLi"], ["Guile", "ChunLi"]) == 3

    # The new target run: {Guile,ChunLi} x {Ryu,Guile,ChunLi} -> max(3,2,6) = 6 -> 63+6 = 69-wide.
    assert _resolve_capacity("auto", ["Guile", "ChunLi"], ["Ryu", "Guile", "ChunLi"]) == 6
    # Explicit int also honoured, and bounds enforced.
    assert _resolve_capacity("6", ["Guile", "ChunLi"], ["Ryu", "Guile", "ChunLi"]) == 6
    for bad in ("8", "10"):
        try:
            _resolve_capacity(bad, ["Guile", "ChunLi"], ["Ryu"])
            raise AssertionError(f"capacity {bad} should have failed the <=7 bound")
        except AssertionError as e:
            if "3-bit" not in str(e) and "> 7" not in str(e):
                raise
    print("test_capacity_math: OK")


def test_width_math():
    """Pure proof of the action-space width (no emulator). The action space is MultiDiscrete([63 +
    n_combo]) with n_combo = combo_capacity if set else len(ego combos); the action_transformer is
    capacity-INDEPENDENT (uses only n_dir, n_att, index), so width is the whole surface."""
    base = 63  # len(DIRECTIONS_BUTTONS)*len(ATTACKS_BUTTONS) = 9*7
    def width(ego, cap):
        n_combo = cap if cap is not None else _macros(ego)
        assert cap is None or _macros(ego) <= cap <= 7
        return base + n_combo
    # legacy per-ego sizing (byte-identical to prior runs)
    assert width("Guile", None) == 66
    assert width("ChunLi", None) == 65
    # capacity == ego's own count == None (padding to own count changes nothing)
    assert width("Guile", 3) == width("Guile", None) == 66
    assert width("ChunLi", 2) == width("ChunLi", None) == 65
    # the generalist roster: capacity 6 -> uniform 69 for BOTH egos
    assert width("Guile", 6) == width("ChunLi", 6) == 69
    print("test_width_math: OK")


# --- Full SFWrapper build. stable-retro allows ONE emulator per process, so each case is built in
#     its OWN subprocess (python test_combo_capacity.py --build EGO OPP CAP -> prints nvec or RAISE:).
def _nvec_subproc(ego, opp, cap):
    import subprocess, sys
    r = subprocess.run([sys.executable, __file__, "--build", ego, opp, ("None" if cap is None else str(cap))],
                       capture_output=True, text=True)
    line = (r.stdout.strip().splitlines() or [""])[-1]
    if not line:
        raise RuntimeError(f"build subproc gave no result (stderr tail): {r.stderr.strip()[-400:]}")
    return line  # "[66]" or "RAISE:<msg>"


def test_sfwrapper_parity():
    # capacity None vs ego's own count -> identical width; and the 69 padding for each ego.
    assert _nvec_subproc("Guile", "ChunLi", None) == "[66]"
    assert _nvec_subproc("Guile", "ChunLi", 3) == "[66]"
    assert _nvec_subproc("ChunLi", "Guile", None) == "[65]"
    assert _nvec_subproc("ChunLi", "Guile", 6) == "[69]"
    assert _nvec_subproc("Guile", "ChunLi", 6) == "[69]"
    # bounds must raise inside SFWrapper: capacity < ego count, and capacity > 7.
    for bad in (1, 8):
        out = _nvec_subproc("Guile", "ChunLi", bad)
        assert out.startswith("RAISE:"), f"capacity={bad} should have raised, got {out}"
    print("test_sfwrapper_parity: OK")


def _build_one(ego_char, opp_char, cap):
    """Build a single SFWrapper in THIS process and print its action_space.nvec (or RAISE:<msg>)."""
    import retro
    from common.retro_wrappers import SFWrapper
    state = f"two_player/{ego_char}_left/Champion.Level1.{ego_char}Vs{opp_char}.2Player.state"
    base = retro.make(game="StreetFighterIISpecialChampionEdition-Genesis", state=state,
                      use_restricted_actions=retro.Actions.FILTERED,
                      obs_type=retro.Observations.IMAGE, players=2)
    try:
        w = SFWrapper(base, side="left", transform_action=True, enable_combo=True,
                      ego_char=ego_char.lower(), left_char=ego_char.lower(), right_char=opp_char.lower(),
                      combo_capacity=cap)
    except AssertionError as e:
        print(f"RAISE:{e}")
        return
    print(list(int(x) for x in w.action_space.nvec))


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and sys.argv[1] == "--build":
        ego, opp, cap = sys.argv[2], sys.argv[3], (None if sys.argv[4] == "None" else int(sys.argv[4]))
        _build_one(ego, opp, cap)
    else:
        test_capacity_math()
        test_width_math()
        test_sfwrapper_parity()
