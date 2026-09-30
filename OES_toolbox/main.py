
import sys

from PyQt6.QtCore import QCommandLineOption, QCommandLineParser
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from OES_toolbox.ui import resources  # seems unused but is needed!

from .cli import parse_cli
from .logger import configure_logging


def main():
    configure_logging()
    app = QApplication(sys.argv)
    app.setApplicationName("OES toolbox")
    parse_cli(app)
    pixmap = QPixmap(":/images/splash.png")
    splash = QSplashScreen(pixmap)
    splash.show()

    from .toolbox import Window
    w = Window()
    splash.finish(w)
    w.show()
    return app.exec()

if __name__ == '__main__':
    sys.exit(main())
    