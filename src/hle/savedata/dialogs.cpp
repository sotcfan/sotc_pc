// libSceCommonDialog, libSceSaveDataDialog and libSceMsgDialog: the system's message boxes.
//
// Nothing is drawn. A dialog with buttons is answered right away, as if the player pressed
// Cross on the button the PS4 would focus (OK, Yes), and its text goes to the log. A dialog
// without buttons (a progress bar, "please wait") stays open until the game closes it, as on
// the PS4. While a dialog is open the game usually pauses itself, so one that never finished
// would freeze the game world.

#include <cstring>
#include <mutex>
#include <string>

#include "common/log.h"
#include "common/types.h"
#include "hle/registry.h"

namespace hle::savedata {

namespace {

constexpr s32 ErrorNotSystemInitialized = static_cast<s32>(0x80B80001);
constexpr s32 ErrorNotInitialized = static_cast<s32>(0x80B80003);
constexpr s32 ErrorAlreadyInitialized = static_cast<s32>(0x80B80004);
constexpr s32 ErrorNotFinished = static_cast<s32>(0x80B80005);
constexpr s32 ErrorInvalidState = static_cast<s32>(0x80B80006);
constexpr s32 ErrorBusy = static_cast<s32>(0x80B80008);
constexpr s32 ErrorNotRunning = static_cast<s32>(0x80B8000B);
constexpr s32 ErrorArgNull = static_cast<s32>(0x80B8000D);
constexpr s32 ErrorNotSupported = static_cast<s32>(0x80B8000F);

enum class Status : u32 { None = 0, Initialized = 1, Running = 2, Finished = 3 };

constexpr u32 ButtonInvalid = 0;
constexpr u32 ButtonFirst = 1; // OK / Yes
constexpr u32 ButtonSecond = 2; // No / Cancel

struct BaseParam {
    u64 size;
    u8 reserved[36];
    u32 magic;
};

std::mutex g_mutex;
bool g_system_initialized = false;
bool g_in_use = false; // a dialog library is initialized

// One dialog library's state.
struct Dialog {
    const char* name;
    Status status = Status::None;
    u32 mode = 0;
    bool waits_for_close = false; // no buttons: the game closes it
    u32 button = ButtonInvalid;

    s32 Initialize() {
        if (!g_system_initialized) {
            return ErrorNotSystemInitialized;
        }
        if (status != Status::None) {
            return ErrorAlreadyInitialized;
        }
        if (g_in_use) {
            return ErrorBusy;
        }
        g_in_use = true;
        status = Status::Initialized;
        return 0;
    }

    s32 Open(u32 new_mode, bool no_buttons, u32 answer) {
        if (status != Status::Initialized && status != Status::Finished) {
            return ErrorInvalidState;
        }
        mode = new_mode;
        waits_for_close = no_buttons;
        button = answer;
        status = Status::Running;
        return 0;
    }

    // The PS4 updates the dialog in the background; an answered one finishes on the next query.
    Status Update() {
        if (status == Status::Running && !waits_for_close) {
            status = Status::Finished;
            LOG_INFO("Dialogs", "{}: answered with button {}", name, button);
        }
        return status;
    }

    s32 Close() {
        if (status != Status::Running) {
            return ErrorNotRunning;
        }
        status = Status::Finished;
        return 0;
    }

    s32 Terminate() {
        if (status == Status::None) {
            return ErrorNotInitialized;
        }
        status = Status::None;
        g_in_use = false;
        return 0;
    }
};

std::string Text(const char* text) {
    return text ? std::string(text) : std::string("(no text)");
}

// ---- libSceCommonDialog ----

PS4_SYSV_ABI s32 sceCommonDialogInitialize() {
    std::scoped_lock lock{g_mutex};
    g_system_initialized = true;
    return 0;
}

PS4_SYSV_ABI bool sceCommonDialogIsUsed() {
    std::scoped_lock lock{g_mutex};
    return g_in_use;
}

// ---- libSceSaveDataDialog ----

namespace save {

enum class Mode : u32 { List = 1, UserMessage = 2, SystemMessage = 3, ErrorCode = 4, ProgressBar = 5 };

struct DirName {
    char data[32];
};

struct Items {
    s32 user_id;
    u32 padding;
    const void* title_id;
    const DirName* dir_names;
    u32 dir_name_count;
    u32 padding2;
    const void* new_item;
    u32 focus;
    u32 padding3;
    const DirName* focus_dir_name;
    u32 item_style;
    u8 reserved[36];
};

struct UserMessage {
    u32 button_type; // 0 OK, 1 Yes/No, 2 none, 3 OK/Cancel
    u32 message_type;
    const char* message;
    u8 reserved[32];
};

struct SystemMessage {
    u32 type; // 1 no data, 2 confirm, 3 overwrite, 4 no space, 5 progress, 6 corrupted, ...
    u32 padding;
    u64 value;
    u8 reserved[32];
};

struct ErrorCode {
    u32 code;
    u8 reserved[32];
};

struct ProgressBar {
    u32 bar_type;
    u32 padding;
    const char* message;
    u32 system_message_type;
    u8 reserved[28];
};

struct Param {
    BaseParam base;
    s32 size;
    Mode mode;
    u32 display_type; // 1 save, 2 load, 3 delete
    u32 padding;
    const void* animation;
    const Items* items;
    const UserMessage* user_message;
    const SystemMessage* system_message;
    const ErrorCode* error_code;
    const ProgressBar* progress_bar;
    void* user_data;
    const void* option;
    u8 reserved[24];
};
static_assert(sizeof(Param) == 0x98);

struct Result {
    u32 mode;
    u32 result; // 0 OK, 1 canceled by the user
    u32 button;
    u32 padding;
    DirName* dir_name;
    void* param;
    void* user_data;
    u8 reserved[32];
};

Dialog g_dialog{"save dialog"};
void* g_user_data = nullptr;
DirName g_chosen{}; // the save picked in a list dialog

PS4_SYSV_ABI s32 sceSaveDataDialogInitialize() {
    std::scoped_lock lock{g_mutex};
    return g_dialog.Initialize();
}

PS4_SYSV_ABI s32 sceSaveDataDialogIsReadyToDisplay() {
    return 1;
}

PS4_SYSV_ABI s32 sceSaveDataDialogOpen(const Param* param) {
    if (!param) {
        return ErrorArgNull;
    }
    std::scoped_lock lock{g_mutex};
    static const char* const DisplayTypes[] = {"?", "save", "load", "delete"};
    const char* display = DisplayTypes[param->display_type < 4 ? param->display_type : 0];
    bool no_buttons = false;
    u32 answer = ButtonFirst;
    g_chosen = {};
    switch (param->mode) {
    case Mode::List: {
        const Items* items = param->items;
        if (items && items->focus_dir_name) {
            g_chosen = *items->focus_dir_name;
        } else if (items && items->dir_names && items->dir_name_count > 0) {
            g_chosen = items->dir_names[0];
        }
        LOG_WARNING("Dialogs", "save dialog ({}): list with {} save(s); chose \"{}\"", display,
                    items ? items->dir_name_count : 0, g_chosen.data);
        break;
    }
    case Mode::UserMessage: {
        const UserMessage* message = param->user_message;
        const u32 buttons = message ? message->button_type : 0;
        no_buttons = buttons == 2;
        LOG_INFO("Dialogs", "save dialog ({}): \"{}\" (buttons {})", display,
                 Text(message ? message->message : nullptr), buttons);
        break;
    }
    case Mode::SystemMessage: {
        const SystemMessage* message = param->system_message;
        const u32 type = message ? message->type : 0;
        no_buttons = type == 5; // "progress"
        LOG_INFO("Dialogs", "save dialog ({}): system message {} (value {})", display, type,
                 message ? message->value : 0);
        break;
    }
    case Mode::ErrorCode:
        LOG_WARNING("Dialogs", "save dialog ({}): error code {:#x}", display,
                    param->error_code ? param->error_code->code : 0);
        break;
    case Mode::ProgressBar:
        no_buttons = true;
        LOG_INFO("Dialogs", "save dialog ({}): progress bar \"{}\"", display,
                 Text(param->progress_bar ? param->progress_bar->message : nullptr));
        break;
    default:
        LOG_WARNING("Dialogs", "save dialog: unknown mode {}", static_cast<u32>(param->mode));
        break;
    }
    g_user_data = param->user_data;
    return g_dialog.Open(static_cast<u32>(param->mode), no_buttons, answer);
}

PS4_SYSV_ABI Status sceSaveDataDialogUpdateStatus() {
    std::scoped_lock lock{g_mutex};
    return g_dialog.Update();
}

PS4_SYSV_ABI Status sceSaveDataDialogGetStatus() {
    std::scoped_lock lock{g_mutex};
    return g_dialog.Update();
}

PS4_SYSV_ABI s32 sceSaveDataDialogGetResult(Result* result) {
    if (!result) {
        return ErrorArgNull;
    }
    std::scoped_lock lock{g_mutex};
    if (g_dialog.status != Status::Finished) {
        return ErrorNotFinished;
    }
    result->mode = g_dialog.mode;
    result->result = 0;
    result->button = g_dialog.button;
    if (result->dir_name) {
        *result->dir_name = g_chosen;
    }
    result->user_data = g_user_data;
    return 0;
}

PS4_SYSV_ABI s32 sceSaveDataDialogProgressBarSetValue(u32 target, u32 rate) {
    std::scoped_lock lock{g_mutex};
    if (g_dialog.status != Status::Running) {
        return ErrorNotRunning;
    }
    if (g_dialog.mode != static_cast<u32>(Mode::ProgressBar) || target != 0) {
        return ErrorNotSupported;
    }
    (void)rate;
    return 0;
}

PS4_SYSV_ABI s32 sceSaveDataDialogClose() {
    std::scoped_lock lock{g_mutex};
    return g_dialog.Close();
}

PS4_SYSV_ABI s32 sceSaveDataDialogTerminate() {
    std::scoped_lock lock{g_mutex};
    return g_dialog.Terminate();
}

} // namespace save

// ---- libSceMsgDialog ----

namespace msg {

enum class Mode : u32 { UserMessage = 1, ProgressBar = 2, SystemMessage = 3 };

// Button layouts of a user message.
enum class Buttons : u32 {
    Ok = 0,
    YesNo = 1,
    None = 2,
    OkCancel = 3,
    Wait = 5,
    WaitCancel = 6,
    YesNoFocusNo = 7,
    OkCancelFocusCancel = 8,
    TwoButtons = 9,
};

struct UserMessage {
    Buttons buttons;
    u32 padding;
    const char* message;
    const void* button_texts;
    u8 reserved[24];
};

struct ProgressBar {
    u32 bar_type;
    u32 padding;
    const char* message;
    u8 reserved[64];
};

struct SystemMessage {
    u32 type;
    u8 reserved[32];
};

struct Param {
    BaseParam base;
    u64 size;
    Mode mode;
    u32 padding;
    const UserMessage* user_message;
    const ProgressBar* progress_bar;
    const SystemMessage* system_message;
    s32 user_id;
    u8 reserved[44];
};
static_assert(sizeof(Param) == 0x88);

struct Result {
    u32 mode;
    u32 result;
    u32 button;
    u8 reserved[32];
};

Dialog g_dialog{"message box"};

PS4_SYSV_ABI s32 sceMsgDialogInitialize() {
    std::scoped_lock lock{g_mutex};
    return g_dialog.Initialize();
}

PS4_SYSV_ABI s32 sceMsgDialogOpen(const Param* param) {
    if (!param) {
        return ErrorArgNull;
    }
    std::scoped_lock lock{g_mutex};
    bool no_buttons = false;
    u32 answer = ButtonFirst;
    switch (param->mode) {
    case Mode::UserMessage: {
        const UserMessage* message = param->user_message;
        const Buttons buttons = message ? message->buttons : Buttons::Ok;
        no_buttons = buttons == Buttons::None || buttons == Buttons::Wait ||
                     buttons == Buttons::WaitCancel;
        if (buttons == Buttons::YesNoFocusNo || buttons == Buttons::OkCancelFocusCancel) {
            answer = ButtonSecond;
        }
        LOG_INFO("Dialogs", "message box: \"{}\" (buttons {})",
                 Text(message ? message->message : nullptr), static_cast<u32>(buttons));
        break;
    }
    case Mode::ProgressBar:
        no_buttons = true;
        LOG_INFO("Dialogs", "message box: progress bar \"{}\"",
                 Text(param->progress_bar ? param->progress_bar->message : nullptr));
        break;
    case Mode::SystemMessage:
        LOG_INFO("Dialogs", "message box: system message {}",
                 param->system_message ? param->system_message->type : 0);
        break;
    default:
        LOG_WARNING("Dialogs", "message box: unknown mode {}", static_cast<u32>(param->mode));
        break;
    }
    return g_dialog.Open(static_cast<u32>(param->mode), no_buttons, answer);
}

PS4_SYSV_ABI Status sceMsgDialogUpdateStatus() {
    std::scoped_lock lock{g_mutex};
    return g_dialog.Update();
}

PS4_SYSV_ABI s32 sceMsgDialogGetResult(Result* result) {
    if (!result) {
        return ErrorArgNull;
    }
    std::scoped_lock lock{g_mutex};
    if (g_dialog.status != Status::Finished) {
        return ErrorNotFinished;
    }
    std::memset(result, 0, sizeof(*result));
    result->mode = g_dialog.mode;
    result->button = g_dialog.button;
    return 0;
}

PS4_SYSV_ABI s32 sceMsgDialogClose() {
    std::scoped_lock lock{g_mutex};
    return g_dialog.Close();
}

PS4_SYSV_ABI s32 sceMsgDialogTerminate() {
    std::scoped_lock lock{g_mutex};
    return g_dialog.Terminate();
}

} // namespace msg

} // namespace

HLE_FUNCTION("libSceCommonDialog", "sceCommonDialogInitialize", sceCommonDialogInitialize);
HLE_FUNCTION("libSceCommonDialog", "sceCommonDialogIsUsed", sceCommonDialogIsUsed);

HLE_FUNCTION("libSceSaveDataDialog", "sceSaveDataDialogInitialize", save::sceSaveDataDialogInitialize);
HLE_FUNCTION("libSceSaveDataDialog", "sceSaveDataDialogIsReadyToDisplay", save::sceSaveDataDialogIsReadyToDisplay);
HLE_FUNCTION("libSceSaveDataDialog", "sceSaveDataDialogOpen", save::sceSaveDataDialogOpen);
HLE_FUNCTION("libSceSaveDataDialog", "sceSaveDataDialogUpdateStatus", save::sceSaveDataDialogUpdateStatus);
HLE_FUNCTION("libSceSaveDataDialog", "sceSaveDataDialogGetStatus", save::sceSaveDataDialogGetStatus);
HLE_FUNCTION("libSceSaveDataDialog", "sceSaveDataDialogGetResult", save::sceSaveDataDialogGetResult);
HLE_FUNCTION("libSceSaveDataDialog", "sceSaveDataDialogProgressBarSetValue", save::sceSaveDataDialogProgressBarSetValue);
HLE_FUNCTION("libSceSaveDataDialog", "sceSaveDataDialogClose", save::sceSaveDataDialogClose);
HLE_FUNCTION("libSceSaveDataDialog", "sceSaveDataDialogTerminate", save::sceSaveDataDialogTerminate);

HLE_FUNCTION("libSceMsgDialog", "sceMsgDialogInitialize", msg::sceMsgDialogInitialize);
HLE_FUNCTION("libSceMsgDialog", "sceMsgDialogOpen", msg::sceMsgDialogOpen);
HLE_FUNCTION("libSceMsgDialog", "sceMsgDialogUpdateStatus", msg::sceMsgDialogUpdateStatus);
HLE_FUNCTION("libSceMsgDialog", "sceMsgDialogGetResult", msg::sceMsgDialogGetResult);
HLE_FUNCTION("libSceMsgDialog", "sceMsgDialogClose", msg::sceMsgDialogClose);
HLE_FUNCTION("libSceMsgDialog", "sceMsgDialogTerminate", msg::sceMsgDialogTerminate);

} // namespace hle::savedata
