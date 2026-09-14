"""
scripted_helpers.py -- deterministic button macros for menus/dialogue/healing.

The RL agent makes the strategic/exploration decisions, but purely mechanical menu
and dialogue steps should never hard-block progress. This module provides a small,
honest toolkit of deterministic helpers that operate directly on PyBoy:

  - ``tap`` / ``hold`` low-level button presses (mirrors the env's press/tick cadence)
  - ``mash_a`` / ``advance_dialogue`` to push through text boxes
  - ``heal_at_counter`` : best-effort "talk to the NPC in front and confirm yes"
                          sequence usable at a Poke Center counter / shop / Nurse
  - ``buy_confirm`` / ``use_confirm`` : confirm-style menu macros

These are intentionally conservative. Navigation *to* the nurse/clerk is left to the
RL policy (or a future scripted navigation skill in ``skills.py``); the helpers here
handle the deterministic "press A through the prompt, choose yes" tail that otherwise
wastes thousands of exploration steps.

``is_text_or_menu_open`` is a heuristic hook: Pokemon Red text/menu state detection
is version-specific, so the default implementation is approximate and clearly marked.
Override or refine it as you validate addresses for your ROM.
"""

from __future__ import annotations

from pyboy.utils import WindowEvent

from curriculum.ram_map import (
    GameState, MOVE_CUT, MOVE_SURF, MOVE_STRENGTH, MOVE_FLY, MOVE_FLASH,
    ITEM_POKE_FLUTE,
)

# Gen-1 move ids that appear in a party Pokemon's overworld field-move submenu. Used
# to compute the cursor offset of a target field move (how many field moves precede it
# in the mon's move slots), so the macro selects the right one regardless of moveset.
FIELD_MOVE_IDS = {
    MOVE_CUT, MOVE_FLY, MOVE_SURF, MOVE_STRENGTH, MOVE_FLASH,
    0x5B,  # DIG
    0x64,  # TELEPORT
    0x87,  # SOFTBOILED
}

# name -> (press_event, release_event)
BUTTONS = {
    "a": (WindowEvent.PRESS_BUTTON_A, WindowEvent.RELEASE_BUTTON_A),
    "b": (WindowEvent.PRESS_BUTTON_B, WindowEvent.RELEASE_BUTTON_B),
    "start": (WindowEvent.PRESS_BUTTON_START, WindowEvent.RELEASE_BUTTON_START),
    "select": (WindowEvent.PRESS_BUTTON_SELECT, WindowEvent.RELEASE_BUTTON_SELECT),
    "up": (WindowEvent.PRESS_ARROW_UP, WindowEvent.RELEASE_ARROW_UP),
    "down": (WindowEvent.PRESS_ARROW_DOWN, WindowEvent.RELEASE_ARROW_DOWN),
    "left": (WindowEvent.PRESS_ARROW_LEFT, WindowEvent.RELEASE_ARROW_LEFT),
    "right": (WindowEvent.PRESS_ARROW_RIGHT, WindowEvent.RELEASE_ARROW_RIGHT),
}

# wd730 holds assorted text/menu control bits; wTextBoxID lives near 0xCC. These
# are approximate "is the game showing a text box / menu" hints for Red. Kept here
# so they are easy to audit and tweak.
ADDR_WD730 = 0xD730
ADDR_TEXT_DELAY_FRAMES = 0xCC25  # nonzero while text is actively scrolling


class ScriptedHelpers:
    """Deterministic emulator macros. One instance per env (shares its PyBoy)."""

    def __init__(self, pyboy, act_freq: int = 24, headless: bool = True,
                 save_video: bool = False):
        self.pyboy = pyboy
        self.act_freq = act_freq
        self.headless = headless
        self.save_video = save_video
        self.gs = GameState(pyboy)

    # ----- low level ------------------------------------------------------- #
    def _render(self) -> bool:
        return self.save_video or not self.headless

    def tap(self, name: str, press_frames: int = 8) -> None:
        """Press a button then release, using the same cadence as the env step."""
        press, release = BUTTONS[name]
        render = self._render()
        self.pyboy.send_input(press)
        self.pyboy.tick(press_frames, render)
        self.pyboy.send_input(release)
        self.pyboy.tick(max(self.act_freq - press_frames - 1, 1), render)
        self.pyboy.tick(1, True)

    def mash_a(self, taps: int = 1) -> None:
        for _ in range(taps):
            self.tap("a")

    # ----- heuristics ------------------------------------------------------ #
    def is_in_battle(self) -> bool:
        return self.gs.in_battle()

    def is_text_or_menu_open(self) -> bool:
        """Approximate: a text box / menu appears active.

        Heuristic and ROM-version-sensitive -- treat as a hint, not ground truth.
        Refine with validated addresses for your ROM if you rely on it heavily.
        """
        try:
            return self.pyboy.memory[ADDR_TEXT_DELAY_FRAMES] != 0
        except Exception:
            return False

    # ----- composite macros ------------------------------------------------ #
    def advance_dialogue(self, max_taps: int = 8) -> int:
        """Tap A up to ``max_taps`` times to push through a dialogue box.

        Stops early once no text box seems open. Returns the number of taps used.
        Safe to call when stuck; it does nothing harmful in the overworld beyond a
        few A presses (which the agent might issue anyway).
        """
        used = 0
        for _ in range(max_taps):
            if not self.is_text_or_menu_open():
                break
            self.tap("a")
            used += 1
        return used

    def heal_at_counter(self, dialogue_taps: int = 16) -> None:
        """Best-effort 'talk to NPC in front and confirm yes'.

        Intended for use once the policy has positioned the player facing a Poke
        Center nurse (or a shop clerk). It taps A to start dialogue, nudges the
        cursor up to the first ("YES"/top) option, confirms, then mashes A through
        the remaining text. Navigation to the counter is the policy's job.
        """
        self.tap("a")                  # initiate dialogue
        self.tap("up")                 # ensure cursor on the top menu option
        self.tap("a")                  # confirm (e.g. "Yes, heal")
        self.mash_a(dialogue_taps)     # flush the rest of the dialogue

    def buy_confirm(self, quantity_taps: int = 0, dialogue_taps: int = 12) -> None:
        """Confirm a purchase: optionally bump quantity up, then accept."""
        self.tap("a")                  # select item / open buy prompt
        for _ in range(quantity_taps):
            self.tap("up")             # increase quantity
        self.tap("a")                  # confirm quantity
        self.tap("a")                  # confirm "Yes"
        self.mash_a(dialogue_taps)

    def use_confirm(self, dialogue_taps: int = 8) -> None:
        """Confirm using/teaching an item (e.g. an HM) through its prompts."""
        self.tap("a")
        self.tap("a")
        self.mash_a(dialogue_taps)

    # ----- field-move: Cut --------------------------------------------------- #
    def field_use_cut(self, dialogue_taps: int = 10) -> bool:
        """Best-effort: open the menu, pick the mon that knows Cut, and use Cut.

        This is the deterministic handling for the single biggest wall just past the
        SS Anne: PPO rarely discovers the multi-step field-move menu sequence on its
        own. We read which party slot knows Cut from RAM so the cursor navigation is
        correct regardless of party order.

        Gen-1 menu path performed:
          START -> (cursor to top) -> POKEMON -> down*slot -> A ->
          (field-move submenu) CUT -> A -> mash A through the cut animation/text.

        Returns False (no-op) when no party member knows Cut. The sequence is
        otherwise self-cancelling if the tile in front is not cuttable (a stray menu
        open/close), so it is safe to call opportunistically. Marked EXPERIMENTAL:
        validate the submenu cursor offset against your ROM before relying on it.
        """
        slot = self.gs.mon_index_with_move(MOVE_CUT)
        if slot < 0:
            return False

        # make sure no menu/text is mid-animation, then open a fresh menu
        self.tap("b")
        self.tap("start")
        # the start menu remembers its last cursor row; push to the very top, then
        # step down to POKEMON. Menu order: POKEDEX(0), POKEMON(1), ITEM, ...
        for _ in range(6):
            self.tap("up")
        self.tap("down")        # POKEDEX -> POKEMON
        self.tap("a")           # open party
        # select the mon that knows Cut
        for _ in range(slot):
            self.tap("down")
        self.tap("a")           # open that mon's action menu
        # the field-move submenu lists usable HM moves at the top; CUT is the only
        # field move most early-game Cut mons have, so it is the top entry.
        self.tap("a")           # choose CUT
        self.mash_a(dialogue_taps)  # flush "used CUT!" text / animation
        return True

    # ----- field-move: generalized (Surf / Strength / Cut / ...) ------------- #
    def field_use_move(self, move_id: int, dialogue_taps: int = 10) -> bool:
        """Best-effort: open the menu, pick the party mon that knows ``move_id`` and
        use that field move on the tile the player faces.

        Unlike :meth:`field_use_cut` (which assumes the field move is the top submenu
        entry), this reads the mon's moveset from RAM and computes the **cursor offset**
        of the target move among the field-usable moves that precede it, so it selects
        the correct entry even when the mon knows several HMs (e.g. a Lapras with both
        Surf and Strength). Returns False when no party member knows the move.

        Gen-1 path: START -> POKEMON -> down*slot -> A -> (field-move submenu) down*offset
        -> A -> mash A through the animation/text. Marked EXPERIMENTAL: validate the
        submenu cursor offsets against your ROM before relying on it for training.
        """
        slot = self.gs.mon_index_with_move(move_id)
        if slot < 0:
            return False
        # cursor offset = number of field-usable moves in earlier move slots
        moves = self.gs.mon_moves(slot)
        offset = 0
        for mv in moves:
            if mv == move_id:
                break
            if mv in FIELD_MOVE_IDS:
                offset += 1

        self.tap("b")
        self.tap("start")
        for _ in range(6):       # force cursor to the top of the start menu
            self.tap("up")
        self.tap("down")         # POKEDEX -> POKEMON
        self.tap("a")            # open party
        for _ in range(slot):    # select the mon that knows the move
            self.tap("down")
        self.tap("a")            # open that mon's field-move submenu
        for _ in range(offset):  # move to the target field move
            self.tap("down")
        self.tap("a")            # choose it
        self.mash_a(dialogue_taps)
        return True

    def field_use_surf(self, dialogue_taps: int = 10) -> bool:
        """Use Surf in the field (must be facing water with Surf usable)."""
        return self.field_use_move(MOVE_SURF, dialogue_taps)

    def field_use_strength(self, dialogue_taps: int = 12) -> bool:
        """Activate Strength (must be facing/near a boulder with Strength usable).

        Strength toggles a 'can push boulders' state for the area; after activation the
        player pushes boulders by simply walking into them. Calling this once when the
        agent is stuck against a boulder is the deterministic handling.
        """
        return self.field_use_move(MOVE_STRENGTH, dialogue_taps)

    # ----- bag item: generalized (Poke Flute / ...) -------------------------- #
    def use_item(self, item_id: int, dialogue_taps: int = 12) -> bool:
        """Open the bag, find ``item_id`` by its current bag index, and USE it.

        Reads the live bag order from RAM so the cursor lands on the right item even as
        the bag contents change. Returns False when the item is not held. EXPERIMENTAL:
        the START-menu ITEM offset assumes the Pokedex is owned (always true by the time
        these key items matter). Validate against your ROM if used earlier.
        """
        items = self.gs.bag_items()
        idx = next((i for i, (iid, _) in enumerate(items) if iid == item_id), -1)
        if idx < 0:
            return False
        self.tap("b")
        self.tap("start")
        for _ in range(6):       # cursor to top
            self.tap("up")
        self.tap("down")         # POKEDEX -> POKEMON
        self.tap("down")         # POKEMON -> ITEM
        self.tap("a")            # open bag
        for _ in range(idx):     # scroll to the item
            self.tap("down")
        self.tap("a")            # select item
        self.tap("a")            # choose USE
        self.mash_a(dialogue_taps)
        return True

    def use_poke_flute(self, dialogue_taps: int = 14) -> bool:
        """Use the Poke Flute (wakes a Snorlax the player is facing)."""
        return self.use_item(ITEM_POKE_FLUTE, dialogue_taps)
