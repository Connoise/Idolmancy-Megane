"""Entry script for the PyInstaller build: launch the Megane GUI."""
import sys

from megane.gui.app import main

if __name__ == "__main__":
    sys.exit(main(sys.argv))
