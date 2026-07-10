# -*- coding: utf-8 -*-
# Copyright 2026 the pyZonae authors
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper so the CLI runs from the repo root without installation.

The actual logic lives in :mod:`pyzonae.cli` (also exposed as the console
script ``pyzonae-classify`` once the package is installed).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyzonae.cli import main

if __name__ == "__main__":
    main()
