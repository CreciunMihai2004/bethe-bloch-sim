"""Application"""

import sys
import ctypes

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from core.data_io import get_base_dir
from ui.main_window import MainWindow


def main():
    if sys.platform == 'win32':
        myappid = 'mycompany.braggsim.app.1.0'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    app = QApplication(sys.argv)
    icon_path = str(get_base_dir() / "Bethe-Bloch-icon.ico")
    app.setWindowIcon(QIcon(icon_path))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()