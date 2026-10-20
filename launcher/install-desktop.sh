#!/bin/sh
# Adds the sotcport launcher to the desktop's application menu (for this user only).
dir=$(dirname "$(readlink -f "$0")")
applications=${XDG_DATA_HOME:-$HOME/.local/share}/applications
mkdir -p "$applications"
cat > "$applications/sotcport-launcher.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Shadow of the Colossus (sotcport)
Comment=Native PC port of Shadow of the Colossus (PS4), with your own copy of the game
Exec="$dir/sotcport-launcher.sh"
Icon=applications-games
Categories=Game;
Terminal=false
StartupWMClass=sotcport-launcher
DESKTOP
echo "Added: $applications/sotcport-launcher.desktop"
