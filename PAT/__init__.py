from PAT.modules import *
import os

with open(os.path.join(os.path.dirname(__file__), 'version.txt'), 'r') as fv:
    __version__ = fv.read().strip()

del os
