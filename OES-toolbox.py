#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from OES_toolbox.main import main
from OES_toolbox.ui import resources  # seems unused but is needed!

if __name__ == '__main__':
    sys.exit(main())