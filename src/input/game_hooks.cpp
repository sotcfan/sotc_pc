// Split actions: telling the game apart two keyboard/mouse inputs that press the same PS4 button.
//
// ==== The problem ===============================================================================
//
// On the PS4 one button does different things depending on the situation. The game reads the
// controller with scePadReadState/scePadRead (src/hle/pad/pad.cpp), which hands it a ScePadData:
// a 32-bit word of buttons (R2 = bit 0x200), the sticks, the triggers. That is all it ever sees.
// So if the mouse's right button and the C key are both bound to R2, the game gets the very same
// bytes from both, and it cannot do "right button grabs, C does the other thing": nothing that
// happens before the game's code (pad.cpp, input.cpp) can change that.
//
// What can: changing the game's code where it *decides* what R2 does, so that at that one place
// it asks us which input pressed the button. That is what the tags in controls.ini are for:
//
//     Mouse Right = R2@Grab
//     C           = R2@Crouch
//
// Both still press R2 (the game works as before, menus included). In addition, input.cpp keeps,
// for every read of the controller, which buttons came from which tag; input::ButtonHeldFor(R2,
// tag) answers "was R2 held *for this purpose*?":
//
//     held by          ButtonHeldFor(R2, Grab)   ButtonHeldFor(R2, Crouch)
//     Mouse Right      true                      false
//     C                false                     true
//     a gamepad        true                      true    (a gamepad is never split)
//     an untagged key  true                      true
//
// A hook (below) then calls the game's function only when the button was meant for it, and
// otherwise tells the game the button is not held.
//
// ==== Where to hook: the two shapes the game's code can have ====================================
//
// A. A table of actions. The game turns buttons into named actions in one place ("grab",
//    "jump"...), and the rest of its code asks for actions, never for buttons: something like
//    bool IsActionHeld(Player*, int action). Then ONE hook is enough: replace that function,
//    and for the action ids that share a button, answer with ButtonHeldFor and the right tag.
//
// B. The button tested where it is used. The player's update reads "R2 held" once and decides in
//    place: if near fur, grab; else if on the ground, the other thing. Then the hook goes on the
//    function of the second thing (or the test that leads to it): when R2 is held only for Grab,
//    skip it (return "not held" / "cannot"), so the game falls through to its "nothing" path.
//
// Which one this game has is not known yet (2026-10-15). The engine is Bluepoint's ("BPE"); the
// map of the executable's source files (analysis notes, mapa_fontes.tsv) puts the engine's input
// code, BPE/Source/Engine/Input/CInputGenerator.cpp, at 0xec5090-0xec6460 (23 functions): a
// good place to start reading.
//
// ==== How to find the function ==================================================================
//
// 1. Where the game reads the controller: start with SOTC_LOG_PAD_CALLERS=1 (pad.cpp); the log
//    says from which addresses of the executable scePadRead/scePadReadState are called and the
//    address of the buffer the game passes.
// 2. In Ghidra (or any disassembler; the executable is loaded at 0x400000, the addresses in the
//    log are the ones Ghidra shows), follow what the caller does with the buffer: it copies the
//    buttons word (offset 0 of ScePadData) into an object of its own. From there, look for
//    tests of the bit 0x200 (R2): "test reg, 0x200" / "and reg, 0x200" / "bt reg, 9".
// 3. Narrow them down while playing: src/diag/call_hooks.h (diag::ObserveCalls) logs calls to a
//    list of functions without changing them. Watch which candidate is called (or returns true)
//    exactly while the button does the action you want to split.
// 4. Write down the address and what the function takes and returns (its arguments in rdi, rsi,
//    rdx, rcx, r8, r9; its result in rax, or xmm0 for floats), fill in the hook below, and try it.
//
// ==== How a hook works ==========================================================================
//
// loader::HookFunction(address, replacement) overwrites the first bytes of the game's function
// with a jump to `replacement` and returns a pointer that runs the original function (its first
// instructions, moved, then a jump back into it). It has to run before Linker::Link, while the
// game's code is still writable: main.cpp calls InstallGameHooks() there.
//
// The replacement must have exactly the same arguments and result as the game's function, and be
// PS4_SYSV_ABI (the game's calling convention: on Windows the compiler would otherwise use
// another one). It runs on the game's thread, in the middle of the game's code: keep it short,
// and never let an exception out of it.

#include "input/game_hooks.h"

#include "common/log.h"
#include "common/types.h"
#include "input/input.h"
#include "loader/hook.h"

namespace input {

namespace {

// ---- Example: R2 split into Grab and Crouch -----------------------------------------------------
//
// A template for a shape-B hook, to be completed once the function is found. Its address is 0:
// nothing is hooked, and the game behaves as without it.
//
// CrouchCheckAddress: the game's function that decides to do the R2 action that is not grabbing.
// The signature below (takes the player object, returns whether to do it) is a guess: replace it
// with the real one found in step 4. The names Grab and Crouch are only the tags of the example
// bindings; any names work, as long as they match controls.ini.
constexpr u64 CrouchCheckAddress = 0; // TODO: fill in (step 4 above)

using CrouchCheckFn = bool PS4_SYSV_ABI (*)(void* player);
CrouchCheckFn g_original_crouch_check = nullptr;
std::uint32_t g_crouch_tag = 0; // bit of the "Crouch" tag (0: no binding uses it)

bool PS4_SYSV_ABI CrouchCheck(void* player) {
    // R2 held, but only by inputs tagged for something else (Mouse Right = R2@Grab): this check
    // must not see it. Gamepads and untagged inputs always pass (see the table at the top).
    if (!ButtonHeldFor(button::R2, g_crouch_tag)) {
        return false;
    }
    return g_original_crouch_check(player);
}

// Hooks the function at `address` (if known) and stores the original in `original`.
template <typename Fn>
void Install(const char* name, u64 address, Fn replacement, Fn& original) {
    if (address == 0) {
        return; // not found in the executable yet
    }
    original = reinterpret_cast<Fn>(
        loader::HookFunction(address, reinterpret_cast<const void*>(replacement)));
    if (original) {
        LOG_INFO("Input", "split action hook: {} at {:#x}", name, address);
    } else {
        LOG_ERROR("Input", "split action hook: could not hook {} at {:#x}", name, address);
    }
}

} // namespace

void InstallGameHooks() {
    g_crouch_tag = TagBit("Crouch");
    // Without the tag in controls.ini there is nothing to split: leave the game's code alone.
    if (g_crouch_tag != 0) {
        Install("crouch check", CrouchCheckAddress, &CrouchCheck, g_original_crouch_check);
    }
}

} // namespace input
