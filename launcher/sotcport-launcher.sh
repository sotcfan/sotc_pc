#!/bin/sh
# Starts the sotcport launcher with a Python that has PySide6. The python3 first in PATH may be a
# pyenv or virtualenv one without it, so the system's is tried too.
dir=$(dirname "$(readlink -f "$0")")
for python in python3 /usr/bin/python3; do
    if "$python" -c "import PySide6.QtWidgets" 2>/dev/null; then
        exec "$python" "$dir/sotcport_launcher.py" "$@"
    fi
done
echo "sotcport launcher: PySide6 not found. Arch/CachyOS: pacman -S pyside6;" \
     "Debian/Ubuntu: apt install python3-pyside6.qtwidgets; elsewhere: pip install PySide6" >&2
exit 1
