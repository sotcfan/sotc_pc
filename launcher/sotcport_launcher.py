#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-2.0-or-later
"""Launcher of sotcport, the native PC port of Shadow of the Colossus (PS4).

Picks the game folder (the player's own dump), edits the port's settings and starts the game with
them, showing its log. The settings reach the game two ways (see OPTIONS.md):
  - sotcport's own options, as environment variables (display, frame rate, GPU layer);
  - the game engine's console variables (ConVars), as its command line after "--": the graphics
    options are the engine's own (shadows, effects, filtering), "+r_motionBlurEnable=0" and so on.
Launcher settings live in ~/.config/sotcport-launcher/settings.json (%APPDATA% on Windows).

Qt (PySide6), one launcher for Linux and Windows. The layout of the settings (choice lists, the
environment built from them, the log page) follows bbport's launcher (launcher/bbport_launcher.py,
GPL-2.0-or-later), rewritten for Qt.
"""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt, QTimer, QUrl
from PySide6.QtGui import QDesktopServices, QFontDatabase, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMainWindow, QPlainTextEdit, QPushButton, QSpinBox, QTabWidget,
    QVBoxLayout, QWidget,
)

from launcher_i18n import UI_LANGUAGES, set_language, tr

LAUNCHER_DIR = Path(__file__).resolve().parent
WINDOWS = sys.platform == "win32"
EXE_NAME = "sotcport.exe" if WINDOWS else "sotcport"
CONFIG_DIR = (Path(os.environ.get("APPDATA", Path.home())) if WINDOWS else
              Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))) / "sotcport-launcher"
CONFIG_FILE = CONFIG_DIR / "settings.json"
DEFAULT_DATA_DIR = (Path(os.environ.get("APPDATA", Path.home())) / "sotcport" if WINDOWS else
                    Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / "sotcport")
MAX_LOG_LINES = 5000

# Choices: (label, value). Labels are translated when shown; the first one is the default
# unless DEFAULTS says otherwise.
PRESENT_MODES = [("Mailbox: no tearing, no waiting (recommended)", "Mailbox"),
                 ("FIFO: waits for the monitor's refresh", "Fifo"),
                 ("FIFO Relaxed: tears when a frame is late", "FifoRelaxed"),
                 ("Immediate: no waiting, tears", "Immediate")]
# The resolution the game renders and outputs at (SOTC_RESOLUTION, src/game/render_resolution.cpp):
# the picture leaves the game at the size it was drawn at, no upscaling. "native" is the
# monitor's (the largest 16:9 size that fits it: the game's camera is 16:9). "" leaves it to the
# game's own PS4 Pro modes (its options menu): the 1080p one draws at 1600x900 and enlarges to
# 1920x1080, the 4K one draws at 2560x1440 and enlarges to 3840x2160.
RESOLUTIONS = [("Native: the monitor's ({})", "native"), ("3840×2160", "3840x2160"),
               ("2560×1440", "2560x1440"), ("1920×1080", "1920x1080"), ("1600×900", "1600x900"),
               ("1280×720", "1280x720"),
               ("The game's menu: the PS4 Pro's modes, enlarged to 1080p or 4K", "")]
# The game's languages (SOTC_LANGUAGE, src/game/language.cpp): the names are the game's own.
# "auto" follows the system's language when the game has it; "" leaves the one chosen in the game
# (asked on its first start and kept in its profile).
GAME_LANGUAGES = [("Automatic: the system's", "auto"), ("The one chosen in the game", ""),
                  ("Português Brasileiro", "pt-BR"), ("Português", "pt"),
                  ("English (US)", "en-US"), ("English (UK)", "en-GB"), ("Español", "es"),
                  ("Español Latino", "es-419"), ("Français", "fr"),
                  ("Français (Canada)", "fr-CA"), ("Deutsch", "de"), ("Italiano", "it"),
                  ("Nederlands", "nl"), ("Dansk", "da"), ("Suomi", "fi"), ("Norsk", "no"),
                  ("Svenska", "sv"), ("Polski", "pl"), ("Русский", "ru"), ("Türkçe", "tr"),
                  ("العربية", "ar")]
FPS_LIMITS = [("30", "30"), ("60", "60"), ("90", "90"), ("120", "120"), ("144", "144"),
              ("165", "165"), ("Uncapped", "uncapped"), ("Custom", "custom")]
WINDOW_SIZES = [("1280×720", "1280x720"), ("1600×900", "1600x900"), ("1920×1080", "1920x1080"),
                ("2560×1440", "2560x1440"), ("3840×2160", "3840x2160")]
SHADOW_DETAIL = [("PS4 Pro (2 cascades)", "2"), ("Cinematic (3 cascades)", "3")]
AO_MODES = [("Baked + screen space (game's default)", "TextureAO_SSAO"),
            ("Baked only", "TextureAO"), ("Screen space only", "SSAO"), ("Off", "Off")]
AA_METHODS = [("Temporal (TAA, game's default)", "TAA"), ("Off", "Off")]
ANISOTROPY = [("Game's default", ""), ("16×", "16")]
TEXTURE_REDUCE = [("Automatic (full above 6 GB of VRAM)", ""), ("Full", "0"),
                  ("Reduced from 2048×2048", "2048"), ("Reduced from 1024×1024", "1024")]
LOG_LEVELS = [("info", "info"), ("warning", "warning"), ("debug", "debug"), ("trace", "trace")]

DEFAULTS = {
    "ui_language": "",
    "game_dir": "",
    "data_dir": "",
    "executable": "",
    # Display
    "fullscreen": True,
    "display": "",
    "window_size": "1280x720",
    "present_mode": "Mailbox",
    "pro_mode": True,
    "fps_limit": "60",
    "fps_custom": 75,
    "resolution": "native",
    "game_language": "auto",
    # Graphics (the engine's ConVars; the values are the game's own defaults on a PS4 Pro)
    "sun_shadows": True,
    "shadow_detail": "2",
    "ao_mode": "TextureAO_SSAO",
    "aa_method": "TAA",
    "motion_blur": True,
    "dof": True,
    "bloom": True,
    "film_grain": True,
    "vignette": True,
    "anisotropy": "",
    "texture_reduce": "",
    # Performance
    "async_shaders": True,
    "draw_pipe": True,
    "async_readback": True,
    "mangohud": False,
    # Advanced
    "extra_env": "",
    "extra_args": "",
    "log_level": "info",
}


def load_settings():
    settings = dict(DEFAULTS)
    try:
        settings.update(json.loads(CONFIG_FILE.read_text()))
    except (OSError, ValueError):
        pass
    return settings


def save_settings(settings):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(settings, indent=2, ensure_ascii=False))
    temporary.replace(CONFIG_FILE)


def find_executable(settings):
    """The sotcport executable: the one chosen, else next to the launcher (a package: sotcport/
    with launcher/ beside it), else in the source tree's build folders (the user's build-clang
    first)."""
    if settings["executable"]:
        return Path(settings["executable"]).expanduser()
    root = LAUNCHER_DIR.parent
    for candidate in (root / EXE_NAME, root / "build-clang" / EXE_NAME, root / "build" / EXE_NAME,
                      root / "build-dev" / EXE_NAME):
        if candidate.is_file():
            return candidate
    return None


def game_problem(path):
    """Why `path` is not a game folder, or None when it is one."""
    if not path:
        return tr("Choose the game folder first")
    if not (Path(path).expanduser() / "eboot.bin").is_file():
        return tr("No eboot.bin in this folder")
    return None


def system_game_language():
    """The game language matching the system's (LANG pt_BR -> pt-BR), or "" if the game has
    none for it."""
    from launcher_i18n import system_language
    code = system_language().replace("_", "-")
    codes = [value for _, value in GAME_LANGUAGES if value not in ("", "auto")]
    if code in codes:
        return code
    if code in ("en", "en-US") or code.startswith("en-"):
        return "en-GB" if code == "en-GB" else "en-US"
    if code.startswith("es-") and code != "es-ES":
        return "es-419"
    base = code.split("-")[0]
    return base if base in codes else ""


def fit_16_9(width, height):
    """The largest 16:9 size, in even numbers, that fits in width x height."""
    if width * 9 >= height * 16:
        return (height * 16 // 9) & ~1, height & ~1
    return width & ~1, (width * 9 // 16) & ~1


def game_environment(s, native="1920x1080"):
    """sotcport's options (OPTIONS.md) as environment variables, from the launcher settings.
    `native`: the monitor's 16:9 size, for the "native" resolution."""
    env = {}
    env["SOTC_FULLSCREEN"] = "1" if s["fullscreen"] else "0"
    if s["display"]:
        env["SOTC_DISPLAY"] = s["display"]
    env["SOTC_WINDOW_SIZE"] = s["window_size"]
    env["SOTC_PRESENT_MODE"] = s["present_mode"]
    if s["pro_mode"]:
        env["SOTC_NEO_MODE"] = "1"
        resolution = native if s["resolution"] == "native" else s["resolution"]
        if resolution:
            env["SOTC_RESOLUTION"] = resolution
        else:
            # A 4K TV, as a PS4 Pro's: the game's menu offers its 4K mode only on one.
            env["SOTC_VIDEO_OUT"] = "3840x2160"
        if s["fps_limit"] == "uncapped":
            env["SOTC_UNCAPPED"] = "1"
        elif s["fps_limit"] == "custom":
            env["SOTC_FPS_LIMIT"] = str(s["fps_custom"])
        else:
            env["SOTC_FPS_LIMIT"] = s["fps_limit"]
    else:
        # On an original PS4 the game's logic breaks above 30 FPS (OPTIONS.md, Frame rate).
        env["SOTC_FPS_LIMIT"] = "30"
    language = system_game_language() if s["game_language"] == "auto" else s["game_language"]
    if language:
        env["SOTC_LANGUAGE"] = language
    if s["texture_reduce"]:
        env["SOTC_TEXTURE_REDUCE"] = s["texture_reduce"]
    if s["async_shaders"]:
        env["SOTC_ASYNC_SHADERS"] = "1"
    if s["draw_pipe"]:
        env["SOTC_DRAW_PIPE"] = "1"
    if s["async_readback"]:
        env["SOTC_ASYNC_READBACK"] = "1"
    if s["mangohud"]:
        env["MANGOHUD"] = "1"  # MangoHud's implicit Vulkan layer turns on with this
    for item in shlex.split(s["extra_env"]):
        if "=" in item:
            key, value = item.split("=", 1)
            env[key] = value
    return env


def game_convars(s):
    """The engine's console variables for the graphics settings, only where they differ from
    the game's defaults (the engine reads "+name=value" from its command line). The names and
    defaults come from the engine's own list (`+help`, 2026-10-20)."""
    args = []
    if not s["sun_shadows"]:
        args.append("+r_enableCascadeShadows=0")
    if s["shadow_detail"] != "2":
        # The game keeps one count per video mode and copies the one of the mode in use; both
        # are set so the PS4 Pro's performance mode takes it too.
        args += [f"+r_virtualGeometryNumShadowCascades={s['shadow_detail']}",
                 f"+r_virtualGeometryNumShadowCascades_Perf={s['shadow_detail']}"]
    if s["ao_mode"] != "TextureAO_SSAO":
        args.append(f"+AO_Mode={s['ao_mode']}")
    if s["aa_method"] != "TAA":
        args.append(f"+r_aaMethod={s['aa_method']}")
    if not s["motion_blur"]:
        args.append("+r_motionBlurEnable=0")
    if not s["dof"]:
        args.append("+r_dofEnable=0")
    if not s["bloom"]:
        args.append("+r_bloomEnable=0")
    if not s["film_grain"]:
        args.append("+filmGrain_enable=0")
    if not s["vignette"]:
        args.append("+r_vignetteScale=0")
    if s["anisotropy"]:
        args.append(f"+linearSamplerAnisoLevel={s['anisotropy']}")
    return args


def connected_displays(executable):
    """The monitors from `sotcport --list-displays`: dicts with the SOTC_DISPLAY value, a label,
    the desktop size and whether it is the primary one; [] if unknown."""
    if not executable:
        return []
    try:
        run = subprocess.run([str(executable), "--list-displays"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    displays = []
    for line in run.stdout.splitlines():
        fields = line.split("\t")
        if len(fields) == 5:
            number, name, size, hertz, primary = fields
            try:
                width, height = (int(x) for x in size.split("x"))
            except ValueError:
                continue
            displays.append({"value": number, "label": f"{number}: {name} ({size}, {hertz} Hz)",
                             "size": (width, height), "primary": primary == "1"})
    return displays


def combo(choices, current):
    """A combo box of (label, value) choices with `current` selected."""
    box = QComboBox()
    for label, value in choices:
        box.addItem(tr(label), value)
    index = box.findData(current)
    box.setCurrentIndex(max(index, 0))
    return box


def note(text):
    """A wrapped, dimmed line under a setting."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setEnabled(False)
    return label


class LauncherWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = load_settings()
        set_language(self.settings["ui_language"])
        self.process = None
        self.setWindowTitle("sotcport — Shadow of the Colossus")
        self.resize(760, 640)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_game_tab(), tr("Game"))
        self.tabs.addTab(self.build_display_tab(), tr("Display"))
        self.tabs.addTab(self.build_graphics_tab(), tr("Graphics"))
        self.tabs.addTab(self.build_performance_tab(), tr("Performance"))
        self.tabs.addTab(self.build_controls_tab(), tr("Controls"))
        self.tabs.addTab(self.build_advanced_tab(), tr("Advanced"))
        self.tabs.addTab(self.build_log_tab(), tr("Log"))

        self.status = QLabel()
        self.play_button = QPushButton(tr("Play"))
        self.play_button.setMinimumHeight(40)
        self.play_button.setMinimumWidth(160)
        font = self.play_button.font()
        font.setPointSize(font.pointSize() + 2)
        font.setBold(True)
        self.play_button.setFont(font)
        self.play_button.clicked.connect(self.on_play)
        bottom = QHBoxLayout()
        bottom.addWidget(self.status, 1)
        bottom.addWidget(self.play_button)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(self.tabs, 1)
        layout.addLayout(bottom)
        self.setCentralWidget(central)
        self.update_state()

    # ---- Pages ----

    def build_game_tab(self):
        page = QWidget()
        form = QFormLayout(page)

        self.game_dir = QLineEdit(self.settings["game_dir"])
        self.game_dir.editingFinished.connect(self.update_state)
        form.addRow(tr("Game folder"), self.folder_row(self.game_dir, tr("Choose the game folder")))
        self.game_status = note("")
        form.addRow("", self.game_status)
        form.addRow("", note(tr("The decrypted dump of Shadow of the Colossus (CUSA08809), the "
                                "folder with eboot.bin. Nothing is written to it.")))

        self.data_dir = QLineEdit(self.settings["data_dir"])
        self.data_dir.setPlaceholderText(str(DEFAULT_DATA_DIR))
        form.addRow(tr("Data folder"), self.folder_row(self.data_dir, tr("Choose the data folder")))
        form.addRow("", note(tr("Saves, the shader cache, screenshots and logs.")))

        self.executable = QLineEdit(self.settings["executable"])
        found = find_executable(dict(self.settings, executable=""))
        self.executable.setPlaceholderText(str(found) if found else "")
        choose = QPushButton(tr("Choose…"))
        choose.clicked.connect(self.on_choose_executable)
        row = QHBoxLayout()
        row.addWidget(self.executable, 1)
        row.addWidget(choose)
        form.addRow(tr("sotcport executable"), row)
        form.addRow("", note(tr("Found automatically next to the launcher or in the build "
                                "folders.")))

        self.game_language = combo(GAME_LANGUAGES, self.settings["game_language"])
        form.addRow(tr("Game language"), self.game_language)
        form.addRow("", note(tr("Menus and subtitles. The game's own choice is asked on its first "
                                "start; a language picked here replaces it on every start.")))

        self.ui_language = combo([(name if code else tr("Default"), code)
                                  for code, name in UI_LANGUAGES], self.settings["ui_language"])
        form.addRow(tr("Launcher language"), self.ui_language)
        form.addRow("", note(tr("Takes effect when the launcher is opened again.")))
        return page

    def build_display_tab(self):
        page = QWidget()
        form = QFormLayout(page)

        self.fullscreen = QCheckBox()
        self.fullscreen.setChecked(self.settings["fullscreen"])
        form.addRow(tr("Fullscreen"), self.fullscreen)
        form.addRow("", note(tr("Fills the monitor at the desktop's resolution (borderless). "
                                "Alt+Enter toggles it in game.")))

        self.display = QComboBox()
        refresh = QPushButton(tr("Refresh"))
        refresh.clicked.connect(self.fill_displays)
        row = QHBoxLayout()
        row.addWidget(self.display, 1)
        row.addWidget(refresh)
        form.addRow(tr("Monitor"), row)
        self.fill_displays()

        self.window_size = combo(WINDOW_SIZES, self.settings["window_size"])
        form.addRow(tr("Window size"), self.window_size)
        self.present_mode = combo(PRESENT_MODES, self.settings["present_mode"])
        form.addRow(tr("VSync"), self.present_mode)

        self.pro_mode = QCheckBox()
        self.pro_mode.setChecked(self.settings["pro_mode"])
        self.pro_mode.toggled.connect(self.update_state)
        form.addRow(tr("PS4 Pro mode"), self.pro_mode)
        form.addRow("", note(tr("The game runs as on a PS4 Pro, where it officially supports 60 "
                                "FPS. Without it the game stays at 30 FPS (above that its logic "
                                "breaks).")))

        self.fps_limit = combo(FPS_LIMITS, self.settings["fps_limit"])
        self.fps_limit.currentIndexChanged.connect(self.update_state)
        self.fps_custom = QSpinBox()
        self.fps_custom.setRange(20, 500)
        self.fps_custom.setValue(int(self.settings["fps_custom"]))
        row = QHBoxLayout()
        row.addWidget(self.fps_limit, 1)
        row.addWidget(self.fps_custom)
        form.addRow(tr("Frame rate limit"), row)
        self.fps_note = note(tr("Only 30 FPS without the PS4 Pro mode"))
        form.addRow("", self.fps_note)

        self.resolution = combo(RESOLUTIONS, self.settings["resolution"])
        form.addRow(tr("Game resolution"), self.resolution)
        form.addRow("", note(tr("The game draws and outputs at this size, with no upscaling. "
                                "Needs the PS4 Pro mode.")))
        return page

    def build_graphics_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(note(tr("Settings of the game's own engine, given to it at start.")))

        shadows = QGroupBox(tr("Shadows"))
        form = QFormLayout(shadows)
        self.sun_shadows = self.check(form, "Sun shadows", "sun_shadows")
        self.shadow_detail = combo(SHADOW_DETAIL, self.settings["shadow_detail"])
        form.addRow(tr("Shadow detail"), self.shadow_detail)
        self.ao_mode = combo(AO_MODES, self.settings["ao_mode"])
        form.addRow(tr("Ambient occlusion"), self.ao_mode)
        layout.addWidget(shadows)

        effects = QGroupBox(tr("Effects"))
        form = QFormLayout(effects)
        self.aa_method = combo(AA_METHODS, self.settings["aa_method"])
        form.addRow(tr("Anti-aliasing"), self.aa_method)
        self.motion_blur = self.check(form, "Motion blur", "motion_blur")
        self.dof = self.check(form, "Depth of field", "dof")
        self.bloom = self.check(form, "Bloom", "bloom")
        self.film_grain = self.check(form, "Film grain", "film_grain")
        self.vignette = self.check(form, "Vignette", "vignette")
        layout.addWidget(effects)

        textures = QGroupBox(tr("Textures"))
        form = QFormLayout(textures)
        self.anisotropy = combo(ANISOTROPY, self.settings["anisotropy"])
        form.addRow(tr("Anisotropic filtering"), self.anisotropy)
        self.texture_reduce = combo(TEXTURE_REDUCE, self.settings["texture_reduce"])
        form.addRow(tr("Texture size in VRAM"), self.texture_reduce)
        form.addRow("", note(tr("Textures that large leave out their finest level: for GPUs "
                                "with little video memory.")))
        layout.addWidget(textures)
        layout.addStretch(1)
        return page

    def build_performance_tab(self):
        page = QWidget()
        form = QFormLayout(page)
        self.async_shaders = self.check(form, "Build shaders in the background", "async_shaders",
                                        "A new area stutters less; an object may show a frame "
                                        "late the first time.")
        self.draw_pipe = self.check(form, "Two-stage draw pipeline", "draw_pipe",
                                    "Decodes and records the GPU commands on two threads.")
        self.async_readback = self.check(form, "Asynchronous readbacks", "async_readback",
                                         "The game's thread waits for the GPU instead of the "
                                         "command thread.")
        self.mangohud = self.check(form, "MangoHud overlay", "mangohud",
                                   "Shows FPS and frame times (MangoHud must be installed).")
        return page

    def build_controls_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(note(tr("Keyboard and mouse bindings live in controls.ini, in the data "
                                 "folder. Controllers work through SDL without setup.")))
        edit = QPushButton(tr("Edit controls.ini"))
        edit.clicked.connect(self.on_edit_controls)
        layout.addWidget(edit, 0, Qt.AlignLeft)
        self.controls_status = note("")
        layout.addWidget(self.controls_status)
        layout.addStretch(1)
        return page

    def build_advanced_tab(self):
        page = QWidget()
        form = QFormLayout(page)
        self.extra_env = QLineEdit(self.settings["extra_env"])
        form.addRow(tr("Extra environment variables"), self.extra_env)
        form.addRow("", note(tr("VARIABLE=value pairs separated by spaces (OPTIONS.md).")))
        self.extra_args = QLineEdit(self.settings["extra_args"])
        form.addRow(tr("Extra game arguments"), self.extra_args)
        form.addRow("", note(tr("The engine's console variables, e.g. +r_vignetteScale=0.5 "
                                "(+help lists them in the log).")))
        self.log_level = combo(LOG_LEVELS, self.settings["log_level"])
        form.addRow(tr("Log level"), self.log_level)
        return page

    def build_log_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(MAX_LOG_LINES)
        self.log.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.log.setLineWrapMode(QPlainTextEdit.NoWrap)
        layout.addWidget(self.log, 1)
        row = QHBoxLayout()
        copy = QPushButton(tr("Copy"))
        copy.clicked.connect(lambda: QApplication.clipboard().setText(self.log.toPlainText()))
        clear = QPushButton(tr("Clear"))
        clear.clicked.connect(self.log.clear)
        row.addStretch(1)
        row.addWidget(copy)
        row.addWidget(clear)
        layout.addLayout(row)
        return page

    # ---- Helpers ----

    def check(self, form, title, key, explanation=None):
        box = QCheckBox()
        box.setChecked(bool(self.settings[key]))
        form.addRow(tr(title), box)
        if explanation:
            form.addRow("", note(tr(explanation)))
        return box

    def folder_row(self, line, title):
        choose = QPushButton(tr("Choose…"))

        def on_choose():
            start = line.text() or line.placeholderText() or str(Path.home())
            folder = QFileDialog.getExistingDirectory(self, title, start)
            if folder:
                line.setText(folder)
                self.update_state()

        choose.clicked.connect(on_choose)
        open_button = QPushButton(tr("Open"))
        open_button.clicked.connect(lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(line.text() or line.placeholderText())))
        row = QHBoxLayout()
        row.addWidget(line, 1)
        row.addWidget(choose)
        row.addWidget(open_button)
        container = QWidget()
        container.setLayout(row)
        row.setContentsMargins(0, 0, 0, 0)
        return container

    def fill_displays(self):
        current = self.display.currentData() if self.display.count() else self.settings["display"]
        self.display.clear()
        self.display.addItem(tr("Automatic"), "")
        self.displays = connected_displays(find_executable(self.store()))
        for display in self.displays:
            self.display.addItem(display["label"], display["value"])
        index = self.display.findData(current)
        self.display.setCurrentIndex(max(index, 0))
        if hasattr(self, "resolution"):
            self.update_state()

    def native_size(self):
        """The 16:9 size that fits the chosen monitor (the primary one for "Automatic"), as
        <w>x<h>; 1920x1080 when the monitors are unknown."""
        chosen = self.display.currentData() if self.display.count() else ""
        displays = getattr(self, "displays", [])
        display = next((d for d in displays if d["value"] == chosen), None) or \
            next((d for d in displays if d["primary"]), None) or \
            (displays[0] if displays else None)
        if not display:
            return "1920x1080"
        width, height = fit_16_9(*display["size"])
        return f"{width}x{height}"

    def store(self):
        """The settings as the pages show them (also saved, when the pages exist)."""
        s = self.settings
        if not hasattr(self, "log_level"):
            return s
        s["game_dir"] = self.game_dir.text().strip()
        s["data_dir"] = self.data_dir.text().strip()
        s["executable"] = self.executable.text().strip()
        s["ui_language"] = self.ui_language.currentData()
        s["game_language"] = self.game_language.currentData()
        s["fullscreen"] = self.fullscreen.isChecked()
        s["display"] = self.display.currentData() or ""
        s["window_size"] = self.window_size.currentData()
        s["present_mode"] = self.present_mode.currentData()
        s["pro_mode"] = self.pro_mode.isChecked()
        s["fps_limit"] = self.fps_limit.currentData()
        s["fps_custom"] = self.fps_custom.value()
        s["resolution"] = self.resolution.currentData()
        s["sun_shadows"] = self.sun_shadows.isChecked()
        s["shadow_detail"] = self.shadow_detail.currentData()
        s["ao_mode"] = self.ao_mode.currentData()
        s["aa_method"] = self.aa_method.currentData()
        s["motion_blur"] = self.motion_blur.isChecked()
        s["dof"] = self.dof.isChecked()
        s["bloom"] = self.bloom.isChecked()
        s["film_grain"] = self.film_grain.isChecked()
        s["vignette"] = self.vignette.isChecked()
        s["anisotropy"] = self.anisotropy.currentData()
        s["texture_reduce"] = self.texture_reduce.currentData()
        s["async_shaders"] = self.async_shaders.isChecked()
        s["draw_pipe"] = self.draw_pipe.isChecked()
        s["async_readback"] = self.async_readback.isChecked()
        s["mangohud"] = self.mangohud.isChecked()
        s["extra_env"] = self.extra_env.text().strip()
        s["extra_args"] = self.extra_args.text().strip()
        s["log_level"] = self.log_level.currentData()
        return s

    def update_state(self):
        s = self.store()
        problem = game_problem(s["game_dir"])
        self.game_status.setText(problem or tr("Game found"))
        executable = find_executable(s)
        running = self.process is not None
        self.fps_limit.setEnabled(self.pro_mode.isChecked())
        self.resolution.setEnabled(self.pro_mode.isChecked())
        self.resolution.setItemText(0, tr("Native: the monitor's ({})").format(
            self.native_size().replace("x", "×")))
        self.fps_custom.setVisible(self.fps_limit.currentData() == "custom")
        self.fps_note.setVisible(not self.pro_mode.isChecked())
        controls = self.data_path() / "controls.ini"
        self.controls_status.setText("" if controls.is_file() else
                                     tr("controls.ini is created the first time the game runs."))
        if running:
            self.status.setText(tr("Running"))
            self.play_button.setText(tr("Stop"))
            self.play_button.setEnabled(True)
        else:
            self.play_button.setText(tr("Play"))
            if problem:
                self.status.setText(problem)
            elif not executable or not executable.is_file():
                self.status.setText(tr("sotcport executable not found"))
            else:
                self.status.setText(tr("Ready"))
            self.play_button.setEnabled(not problem and executable is not None)

    def data_path(self):
        return Path(self.data_dir.text().strip() or DEFAULT_DATA_DIR).expanduser()

    def on_choose_executable(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("Choose the sotcport executable"),
                                              str(LAUNCHER_DIR.parent))
        if path:
            self.executable.setText(path)
            self.update_state()

    def on_edit_controls(self):
        controls = self.data_path() / "controls.ini"
        if controls.is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(controls)))

    # ---- The game ----

    def on_play(self):
        if self.process is not None:
            self.stop_game()
        else:
            self.start_game()

    def start_game(self):
        s = self.store()
        save_settings(s)
        executable = find_executable(s)
        arguments = [str(Path(s["game_dir"]).expanduser())]
        if s["data_dir"]:
            arguments += ["--data", str(Path(s["data_dir"]).expanduser())]
        arguments += ["--log", s["log_level"]]
        game_args = game_convars(s) + shlex.split(s["extra_args"])
        if game_args:
            arguments += ["--"] + game_args

        environment = QProcessEnvironment.systemEnvironment()
        for key, value in game_environment(s, self.native_size()).items():
            environment.insert(key, value)

        self.log.clear()
        settings_env = " ".join(f"{k}={v}" for k, v in game_environment(s, self.native_size()).items())
        self.append_log(f"$ {settings_env} {executable} {' '.join(arguments)}\n")
        self.process = QProcess(self)
        self.process.setProcessEnvironment(environment)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.on_output)
        self.process.finished.connect(self.on_finished)
        self.process.errorOccurred.connect(self.on_error)
        self.process.start(str(executable), arguments)
        self.update_state()

    def stop_game(self):
        if self.process is None:
            return
        self.process.terminate()
        # The game saves its screenshots and closes on its own; a hung one is killed.
        QTimer.singleShot(8000, lambda p=self.process: p and p.state() != QProcess.NotRunning
                          and p.kill())

    def on_output(self):
        if self.process is not None:
            data = self.process.readAllStandardOutput().data().decode("utf-8", "replace")
            self.append_log(data)

    def append_log(self, text):
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def on_finished(self, code, _status):
        self.on_output()
        self.append_log("\n" + tr("The game closed (exit status {})").format(code) + "\n")
        self.process = None
        self.update_state()

    def on_error(self, error):
        if error == QProcess.FailedToStart:
            self.append_log(f"\n{self.process.errorString()}\n")
            self.process = None
            self.update_state()

    def closeEvent(self, event):
        save_settings(self.store())
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("sotcport-launcher")
    app.setDesktopFileName("sotcport-launcher")
    window = LauncherWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
