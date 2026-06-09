"""GUI application entry point: ``megane gui`` / ``python -m megane.gui``."""
from __future__ import annotations

import sys

from .ng_compat import QtWidgets
from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(argv)
    app.setApplicationName("Megane")
    window = MainWindow()
    # `megane gui project.megane` opens the project directly.
    file_args = [a for a in argv[1:] if not a.startswith("-")]
    if file_args:
        window.load_project(file_args[0])
    window.show()
    return app.exec()
