# SPDX-License-Identifier: GPL-2.0-or-later
"""Translations of the sotcport launcher.

The English text in the launcher's code is the key; LANGUAGES maps a language code to its
translations. A text missing from a language shows in English. The idea of keying translations by
the source text comes from bbport's launcher (launcher/bbport_i18n.py, GPL-2.0-or-later).
"""

import locale
import os

# (code, name shown in the language menu). "" follows the system.
UI_LANGUAGES = [("", "System"), ("en", "English"), ("pt_BR", "Português (Brasil)")]

PT_BR = {
    # Window and buttons
    "Play": "Jogar",
    "Stop": "Parar",
    "Running": "Rodando",
    "Ready": "Pronto",
    "The game closed (exit status {})": "O jogo fechou (código de saída {})",
    "The game is already running": "O jogo já está aberto",
    "Choose…": "Escolher…",
    "Open": "Abrir",
    "Refresh": "Atualizar",
    "Default": "Padrão",
    "Automatic": "Automático",
    "On": "Ligado",
    "Off": "Desligado",
    # Tabs
    "Game": "Jogo",
    "Display": "Tela",
    "Graphics": "Gráficos",
    "Performance": "Desempenho",
    "Controls": "Controles",
    "Advanced": "Avançado",
    "Log": "Log",
    # Game tab
    "Game folder": "Pasta do jogo",
    "The decrypted dump of Shadow of the Colossus (CUSA08809), the folder with eboot.bin. "
    "Nothing is written to it.":
        "O dump descriptografado de Shadow of the Colossus (CUSA08809), a pasta com o eboot.bin. "
        "Nada é escrito nela.",
    "Data folder": "Pasta de dados",
    "Saves, the shader cache, screenshots and logs.":
        "Saves, cache de shaders, capturas de tela e logs.",
    "sotcport executable": "Executável do sotcport",
    "Found automatically next to the launcher or in the build folders.":
        "Encontrado sozinho ao lado do launcher ou nas pastas de build.",
    "Choose the game folder": "Escolha a pasta do jogo",
    "Choose the data folder": "Escolha a pasta de dados",
    "Choose the sotcport executable": "Escolha o executável do sotcport",
    "No eboot.bin in this folder": "Não há eboot.bin nesta pasta",
    "Game found": "Jogo encontrado",
    "Choose the game folder first": "Escolha a pasta do jogo primeiro",
    "sotcport executable not found": "Executável do sotcport não encontrado",
    "Launcher language": "Idioma do launcher",
    "Game language": "Idioma do jogo",
    "Automatic: the system's": "Automático: o do sistema",
    "The one chosen in the game": "O escolhido no jogo",
    "Menus and subtitles. The game's own choice is asked on its first start; a language picked "
    "here replaces it on every start.":
        "Menus e legendas. O jogo pergunta o idioma na primeira vez que abre; um idioma escolhido "
        "aqui vale em todas as aberturas no lugar desse.",
    "Takes effect when the launcher is opened again.": "Vale ao abrir o launcher de novo.",
    # Display tab
    "Fullscreen": "Tela cheia",
    "Fills the monitor at the desktop's resolution (borderless). Alt+Enter toggles it in "
    "game.":
        "Ocupa o monitor na resolução da área de trabalho (sem borda). Alt+Enter alterna "
        "durante o jogo.",
    "Monitor": "Monitor",
    "Window size": "Tamanho da janela",
    "VSync": "VSync",
    "Mailbox: no tearing, no waiting (recommended)": "Mailbox: sem rasgo, sem espera (recomendado)",
    "FIFO: waits for the monitor's refresh": "FIFO: espera a atualização do monitor",
    "FIFO Relaxed: tears when a frame is late": "FIFO Relaxed: rasga quando um quadro atrasa",
    "Immediate: no waiting, tears": "Immediate: sem espera, com rasgo",
    "PS4 Pro mode": "Modo PS4 Pro",
    "The game runs as on a PS4 Pro, where it officially supports 60 FPS. Without it the game "
    "stays at 30 FPS (above that its logic breaks).":
        "O jogo roda como num PS4 Pro, onde ele suporta 60 FPS oficialmente. Sem isso o jogo "
        "fica em 30 FPS (acima disso a lógica quebra).",
    "Frame rate limit": "Limite de FPS",
    "Uncapped": "Sem limite",
    "Custom": "Personalizado",
    "Only 30 FPS without the PS4 Pro mode": "Só 30 FPS sem o modo PS4 Pro",
    "Game resolution": "Resolução do jogo",
    "Native: the monitor's ({})": "Nativa: a do monitor ({})",
    "The game's menu: the PS4 Pro's modes, enlarged to 1080p or 4K":
        "A do menu do jogo: os modos do PS4 Pro, ampliados para 1080p ou 4K",
    "The game draws and outputs at this size, with no upscaling. Needs the PS4 Pro mode.":
        "O jogo desenha e entrega a imagem neste tamanho, sem upscaling. Precisa do modo PS4 "
        "Pro.",
    # Graphics tab
    "Settings of the game's own engine, given to it at start.":
        "Configurações do próprio motor do jogo, passadas a ele ao iniciar.",
    "Shadows": "Sombras",
    "Sun shadows": "Sombras do sol",
    "Shadow detail": "Detalhe das sombras",
    "PS4 Pro (2 cascades)": "PS4 Pro (2 cascatas)",
    "Cinematic (3 cascades)": "Cinematográfico (3 cascatas)",
    "Ambient occlusion": "Oclusão de ambiente",
    "Baked + screen space (game's default)": "Pré-calculada + na tela (padrão do jogo)",
    "Baked only": "Só pré-calculada",
    "Screen space only": "Só na tela",
    "Effects": "Efeitos",
    "Anti-aliasing": "Anti-serrilhado",
    "Temporal (TAA, game's default)": "Temporal (TAA, padrão do jogo)",
    "Motion blur": "Desfoque de movimento",
    "Depth of field": "Profundidade de campo",
    "Bloom": "Bloom",
    "Film grain": "Granulação de filme",
    "Vignette": "Vinheta",
    "Textures": "Texturas",
    "Anisotropic filtering": "Filtragem anisotrópica",
    "Game's default": "Padrão do jogo",
    "Texture size in VRAM": "Tamanho das texturas na VRAM",
    "Automatic (full above 6 GB of VRAM)": "Automático (inteiras acima de 6 GB de VRAM)",
    "Full": "Inteiras",
    "Reduced from 2048×2048": "Reduzidas a partir de 2048×2048",
    "Reduced from 1024×1024": "Reduzidas a partir de 1024×1024",
    "Textures that large leave out their finest level: for GPUs with little video memory.":
        "Texturas desse tamanho deixam de fora o nível mais fino: para placas com pouca "
        "memória de vídeo.",
    # Performance tab
    "Build shaders in the background": "Montar shaders em segundo plano",
    "A new area stutters less; an object may show a frame late the first time.":
        "Uma área nova engasga menos; um objeto pode aparecer um quadro atrasado na primeira "
        "vez.",
    "Two-stage draw pipeline": "Pipeline de desenho em dois estágios",
    "Decodes and records the GPU commands on two threads.":
        "Decodifica e grava os comandos da GPU em duas threads.",
    "Asynchronous readbacks": "Leituras da GPU assíncronas",
    "The game's thread waits for the GPU instead of the command thread.":
        "A thread do jogo espera a GPU no lugar da thread de comandos.",
    "MangoHud overlay": "Overlay do MangoHud",
    "Shows FPS and frame times (MangoHud must be installed).":
        "Mostra FPS e tempos de quadro (o MangoHud precisa estar instalado).",
    # Controls tab
    "Keyboard and mouse bindings live in controls.ini, in the data folder. Controllers work "
    "through SDL without setup.":
        "As teclas e o mouse ficam no controls.ini, na pasta de dados. Controles funcionam pelo "
        "SDL sem configurar.",
    "Edit controls.ini": "Editar controls.ini",
    "controls.ini is created the first time the game runs.":
        "O controls.ini é criado na primeira vez que o jogo roda.",
    # Advanced tab
    "Extra environment variables": "Variáveis de ambiente extras",
    "VARIABLE=value pairs separated by spaces (OPTIONS.md).":
        "Pares VARIAVEL=valor separados por espaço (OPTIONS.md).",
    "Extra game arguments": "Argumentos extras do jogo",
    "The engine's console variables, e.g. +r_vignetteScale=0.5 (+help lists them in the log).":
        "Variáveis de console do motor, por exemplo +r_vignetteScale=0.5 (+help lista todas "
        "no log).",
    "Log level": "Nível do log",
    # Log tab
    "Copy": "Copiar",
    "Clear": "Limpar",
}

LANGUAGES = {"pt_BR": PT_BR}

_current = {}


def system_language():
    """The system's language code (pt_BR, en_US...), from the environment like Qt does."""
    for variable in ("LC_ALL", "LC_MESSAGES", "LANG"):
        value = os.environ.get(variable, "")
        if value and value not in ("C", "POSIX"):
            return value.split(".")[0]
    try:
        return (locale.getlocale()[0] or "en").split(".")[0]
    except ValueError:
        return "en"


def set_language(code):
    """code: one of UI_LANGUAGES ("" = the system's; pt_PT and the like fall back to pt_BR)."""
    global _current
    code = code or system_language()
    if code not in LANGUAGES:
        code = next((key for key in LANGUAGES if key.split("_")[0] == code.split("_")[0]), "en")
    _current = LANGUAGES.get(code, {})


def tr(text):
    return _current.get(text, text)
