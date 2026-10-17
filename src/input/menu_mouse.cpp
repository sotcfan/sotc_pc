// The mouse in the game's menus: a cursor that selects what it points at, confirms with the left
// button (and the right button goes back, the wheel moves the selection: input.cpp).
//
// ==== The game's menus (found 2026-10-16 with diag/data_watch.h and diag/camera_probe.cpp) =====
//
// The menus are the engine's UI (BPE, data-driven): a menu is a "menu grid" behavior
// (CComponentUIBehaviorMenuGrid; its loader CCPLdrUIBehaviorMenuGrid is at 0x720bb0 and its code
// follows it) on a UI node, with the items as UI nodes. The grid's input handling runs at every
// update while the menu has the input:
//
//   0x722780(grid, update)    update[0] = the grid's UI node, update[2] = a context; reads the
//                             directional buttons through the engine's action array (0x63fb60:
//                             down, 0x63fa80: up, 0x63f9a0 / 0x63f8c0: left / right, with repeat)
//                             and moves the selection; returns 1 when it handled something
//   0x7231e0(grid, node, context, step, from, 0)
//                             moves the selection `step` items from item `from`, skipping
//                             disabled items; the grid itself uses (1, k - 1) to select item k
//
// The grid (offsets from its address):
//   +0xa0     its configuration; +0xec there: 0 = horizontal, 1 = vertical, 2 = no navigation
//   +0xf0     the items: a vector (begin, end) of UI node pointers
//   +0x190    the selected item's index
//
// A UI node (the menu items, and everything on screen):
//   +0x1b0    parent node (null at the root, the 1920 x 1080 screen)
//   +0x1d8    state flags (2: disabled, as the grid tests them; the game ORs the parents' in)
//   +0x1e4    position (x, y), +0x1ec size (w, h), +0x1f4 position offset, +0x1fc size offset:
//             the node's rectangle in its parent is position + offset, size + size offset, and
//             its place on screen the sum of position + offset up the parents (0x712ff0)
//
// An item node is only a small marker; what the player sees as "the item" is the row it sits
// in: the item's ancestor that is a direct child of the grid's node (in the pause menu, a node
// 1334 x 40 across the menu's width, 45 apart). That row is what the cursor is tested against.
// In rows that hold a value (options), the item node is the value control instead (see
// ValueControlMinWidth).
//
// The menu is open while 0x722780 keeps being called (NoteMenuActive): the mouse is then a cursor.
// The world map (below) counts as a menu too.

#include <array>

#include "common/log.h"
#include "common/types.h"
#include "input/game_hooks.h"
#include "input/input.h"
#include "loader/hook.h"

namespace input {

namespace {

constexpr u64 MenuGridInputAddress = 0x722780;
constexpr u64 MenuGridSelectAddress = 0x7231e0;

constexpr u32 GridConfigOffset = 0xa0;
constexpr u32 GridItemsOffset = 0xf0;
constexpr u32 GridSelectedOffset = 0x190;

constexpr u32 NodeParentOffset = 0x1b0;
constexpr u32 NodeFlagsOffset = 0x1d8;
constexpr u32 NodePositionOffset = 0x1e4;
constexpr u32 NodeSizeOffset = 0x1ec;
constexpr u32 NodePositionOffsetOffset = 0x1f4;
constexpr u32 NodeSizeOffsetOffset = 0x1fc;
constexpr u32 NodeDisabledFlag = 0x2;

// Value rows ("< Performance Mode >", sliders): there the item node is the control itself
// (300 x 36 in the Display settings), while elsewhere it is a 5 x 5 marker. Clicking its left
// half holds Left, its right half Right, for as long as the mouse button is held (sliders read
// the direction held: 0x75a790); the arrows are drawn a little outside the node.
constexpr float ValueControlMinWidth = 20.0f;
constexpr float ValueArrowMargin = 40.0f;

using MenuGridInputFn = u32 PS4_SYSV_ABI (*)(u8* grid, u64* update);
using MenuGridSelectFn = bool PS4_SYSV_ABI (*)(u8* grid, u64 node, u64 context, s32 step,
                                               s32 from, s32 flag);
MenuGridInputFn g_original_input = nullptr;
const auto g_select = reinterpret_cast<MenuGridSelectFn>(MenuGridSelectAddress);

// The sound of a selection moving. When the directional buttons move it, the grid's input
// handling (0x722a9d) evaluates its configuration's "move sound" (+0x100, an evaluator: virtual
// +0x38, with an evaluation context made by 0x7ca820(context, grid, game context), 0x48 bytes)
// and plays what it names: 0xe5fb70(*0x36f8ff0, name, 0). The selection moved by the mouse
// plays it the same way; selecting through 0x7231e0 alone (as the grid does for selections that
// come from its data) is silent.
constexpr u64 EvaluationContextAddress = 0x7ca820;
constexpr u64 PlaySoundAddress = 0xe5fb70;
constexpr u64 SoundPlayerAddress = 0x36f8ff0; // a global holding the player of UI sounds
constexpr u32 ConfigMoveSoundOffset = 0x100;
constexpr u32 EvaluateMethodOffset = 0x38;

using EvaluationContextFn = void* PS4_SYSV_ABI (*)(void* context, u8* grid, u64 game_context);
using EvaluateFn = const u64* PS4_SYSV_ABI (*)(u8* evaluator, void* context);
using PlaySoundFn = void PS4_SYSV_ABI (*)(u64 player, u64 name, s32 flag);

void PlayMoveSound(u8* grid, u64 game_context) {
    const u8* config = *reinterpret_cast<const u8* const*>(grid + GridConfigOffset);
    u8* evaluator = config ? *reinterpret_cast<u8* const*>(config + ConfigMoveSoundOffset)
                           : nullptr;
    if (!evaluator) {
        return; // this menu has no sound for it
    }
    alignas(16) u8 context[0x48] = {};
    reinterpret_cast<EvaluationContextFn>(EvaluationContextAddress)(context, grid, game_context);
    const u8* vtable = *reinterpret_cast<const u8* const*>(evaluator);
    const auto evaluate = *reinterpret_cast<const EvaluateFn*>(vtable + EvaluateMethodOffset);
    const u64* name = evaluate(evaluator, context);
    if (!name) {
        return;
    }
    reinterpret_cast<PlaySoundFn>(PlaySoundAddress)(
        *reinterpret_cast<const u64*>(SoundPlayerAddress), *name, 0);
}

template <typename T>
T Get(const u8* base, u32 offset) {
    return *reinterpret_cast<const T*>(base + offset);
}

const u8* Parent(const u8* node) {
    return Get<const u8*>(node, NodeParentOffset);
}

struct Rect {
    float x0, y0, x1, y1;
    bool Contains(float x, float y) const {
        return x >= x0 && x < x1 && y >= y0 && y < y1;
    }
};

// The node's rectangle on screen (UI coordinates).
Rect ScreenRect(const u8* node) {
    float x = 0, y = 0;
    for (const u8* n = node; n; n = Parent(n)) {
        x += Get<float>(n, NodePositionOffset) + Get<float>(n, NodePositionOffsetOffset);
        y += Get<float>(n, NodePositionOffset + 4) + Get<float>(n, NodePositionOffsetOffset + 4);
    }
    const float w = Get<float>(node, NodeSizeOffset) + Get<float>(node, NodeSizeOffsetOffset);
    const float h = Get<float>(node, NodeSizeOffset + 4) + Get<float>(node, NodeSizeOffsetOffset + 4);
    return {x, y, x + w, y + h};
}

// The item's row: its ancestor right under the grid's node (the item itself if it is not under
// it), and whether the item or anything up to the grid is disabled.
const u8* Row(const u8* item, const u8* grid_node, bool& disabled) {
    const u8* row = item;
    u32 flags = Get<u32>(item, NodeFlagsOffset);
    while (Parent(row) && Parent(row) != grid_node) {
        row = Parent(row);
        flags |= Get<u32>(row, NodeFlagsOffset);
    }
    disabled = (flags & NodeDisabledFlag) != 0;
    return Parent(row) == grid_node ? row : item;
}

// The cursor movements each grid has already followed (a few grids can be open at once).
struct Seen {
    const u8* grid = nullptr;
    std::uint32_t moves = 0;
};
std::array<Seen, 8> g_seen{};

// A grid seen for the first time starts as having followed every movement so far: a menu that
// just opened keeps its own first selection until the cursor moves (it was opened by a click,
// and the cursor still sits where that click was).
std::uint32_t& SeenMoves(const u8* grid, std::uint32_t moves) {
    for (Seen& seen : g_seen) {
        if (seen.grid == grid) {
            return seen.moves;
        }
    }
    // A grid not seen lately: take the oldest slot (simply rotate).
    static u32 next = 0;
    Seen& slot = g_seen[next++ % g_seen.size()];
    slot = {grid, moves};
    return slot.moves;
}

u32 PS4_SYSV_ABI MenuGridInput(u8* grid, u64* update) {
    const u32 result = g_original_input(grid, update);
    NoteMenuActive();
    float x, y;
    std::uint32_t moves;
    const bool cursor = MenuCursor(x, y, moves);
    float click_x, click_y;
    const bool click = PeekMenuClick(click_x, click_y);
    if (!cursor && !click) {
        return result;
    }
    const auto* grid_node = reinterpret_cast<const u8*>(update[0]);
    const auto* begin = Get<u8* const*>(grid, GridItemsOffset);
    const auto* end = Get<u8* const*>(grid, GridItemsOffset + 8);
    if (!begin || end <= begin || end - begin > 256) {
        return result;
    }
    // The item under a point, -1 for none.
    const auto item_at = [&](float px, float py) {
        for (auto it = begin; it != end; ++it) {
            bool disabled;
            const u8* row = Row(*it, grid_node, disabled);
            if (!disabled && ScreenRect(row).Contains(px, py)) {
                return static_cast<s32>(it - begin);
            }
        }
        return -1;
    };
    const s32 selected = Get<s32>(grid, GridSelectedOffset);
    const auto select = [&](s32 index) {
        if (index != Get<s32>(grid, GridSelectedOffset) &&
            g_select(grid, update[0], update[2], 1, index - 1, 0)) {
            PlayMoveSound(grid, update[2]);
        }
    };
    // Pointing: only when the cursor moved, so a cursor left over an item does not take the
    // selection back from the keyboard or a controller.
    std::uint32_t& seen = SeenMoves(grid, cursor ? moves : 0);
    if (cursor && seen != moves) {
        seen = moves;
        const s32 hovered = item_at(x, y);
        if (hovered >= 0 && hovered != selected) {
            select(hovered);
        }
    }
    // A click on an item: select it (if pointing has not) and press Cross, as the controller
    // would; on a value control, step the value instead. Clicks beside the items are left for
    // another grid, or expire.
    if (click) {
        const s32 clicked = item_at(click_x, click_y);
        if (clicked >= 0) {
            TakeMenuClick();
            select(clicked);
            const u8* item = begin[clicked];
            if (Get<float>(item, NodeSizeOffset) + Get<float>(item, NodeSizeOffsetOffset) >
                ValueControlMinWidth) {
                Rect control = ScreenRect(item);
                control.x0 -= ValueArrowMargin;
                control.x1 += ValueArrowMargin;
                if (control.Contains(click_x, click_y)) {
                    // Held while the mouse button is: a click steps a list of options once,
                    // holding runs a slider (which reads the direction held, not pressed).
                    HoldButtonWhileClicked(click_x < (control.x0 + control.x1) / 2
                                               ? button::Left
                                               : button::Right);
                }
            } else {
                PressButtonOnce(button::Cross);
            }
        }
    }
    return result;
}

// ---- The world map -------------------------------------------------------------------------
//
// The map (CComponentUIBehaviorShadowWorldMap; loader at 0x8b2730, code after it) is not a menu
// grid. Its update 0x8b4c20(map, update) moves the map with the left stick (actions 154/155 of
// the action array at update[3]) and keeps where the map is in +0xb8 / +0xbc (UI units), clamped
// to the map's edges in the same update. While the map is open the mouse is a cursor, and
// dragging with the left button moves the map with the cursor: the cursor's movement is added
// to +0xb8 / +0xbc after the update (the next update clamps it).
constexpr u64 WorldMapUpdateAddress = 0x8b4c20;
constexpr u32 MapOffsetOffset = 0xb8;

using WorldMapUpdateFn = u32 PS4_SYSV_ABI (*)(u8* map, u64* update);
WorldMapUpdateFn g_original_map = nullptr;

u32 PS4_SYSV_ABI WorldMapUpdate(u8* map, u64* update) {
    const u32 result = g_original_map(map, update);
    NoteMenuActive(true);
    static float last_x = -1, last_y = -1;
    float x, y;
    std::uint32_t moves;
    if (!MenuDragging() || !MenuCursor(x, y, moves)) {
        last_x = last_y = -1;
        return result;
    }
    if (last_x >= 0) {
        auto* offset = reinterpret_cast<float*>(map + MapOffsetOffset);
        offset[0] += x - last_x;
        offset[1] += y - last_y;
    }
    last_x = x;
    last_y = y;
    return result;
}

} // namespace

void InstallMenuMouse() {
    g_original_map = reinterpret_cast<WorldMapUpdateFn>(loader::HookFunction(
        WorldMapUpdateAddress, reinterpret_cast<const void*>(&WorldMapUpdate)));
    if (!g_original_map) {
        LOG_ERROR("Input", "menu mouse: could not hook the map at {:#x}", WorldMapUpdateAddress);
    }
    g_original_input = reinterpret_cast<MenuGridInputFn>(loader::HookFunction(
        MenuGridInputAddress, reinterpret_cast<const void*>(&MenuGridInput)));
    if (!g_original_input) {
        LOG_ERROR("Input", "menu mouse: could not hook the menus at {:#x}", MenuGridInputAddress);
    }
}

} // namespace input
