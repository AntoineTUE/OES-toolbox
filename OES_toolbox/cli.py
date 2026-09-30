import pyqtgraph as pg
from PyQt6.QtCore import QCommandLineOption, QCommandLineParser, Qt
from PyQt6.QtWidgets import QApplication, QStyleFactory

from ._version import version


def parse_cli(app: QApplication):
    themes = QStyleFactory.keys()
    _name = app.applicationName()

    parser = QCommandLineParser()
    parser.setApplicationDescription("Tool for helping with OES in plasma physics.")
    app.setApplicationVersion(f"{version}")

    parser.addHelpOption()
    parser.addVersionOption()
    style_option = QCommandLineOption(["t", "theme"], "Use either of the themes: " + ", ".join(themes), "theme")
    darkMode_option = QCommandLineOption(
        ["colorscheme"],
        "Set application to <dark> or <light> mode (or follow system <native> theme)",
        "dark_or_light",
        "native",
    )
    parser.addOption(style_option)
    parser.addOption(darkMode_option)
    parser.process(app)

    kwargs = {}
    
    match parser.value(darkMode_option):
        case "light":
            app.styleHints().setColorScheme(Qt.ColorScheme.Light)
            pg.setConfigOption("background", "#ffffff33")
            pg.setConfigOption("foreground","k")
        case "dark":
            app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
            pg.setConfigOption("background", "#00000011")
            pg.setConfigOption("foreground", "#eeeeee")
        case _: # native, adopt something neutral between light and dark
            pg.setConfigOption("background","#ffffff00") # no bg
            pg.setConfigOption("foreground", "#7C7C7C")
    app.style().unpolish(app)
    if parser.isSet(style_option):
        app.setStyle(parser.value(style_option))
        
    else:
        app.setStyle("windowsvista" if parser.value(darkMode_option)!="dark" else "fusion")
    app.setPalette(app.palette())
    app.style().polish(app)

    return kwargs
