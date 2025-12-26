#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, runpy

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

RAW = os.path.join(BASE_DIR, "Codigo_Factorizado_raw.py")

if not os.path.exists(RAW):
    print("ERROR: No se encontró 'Codigo_Factorizado_raw.py'.")
    sys.exit(1)

try:
    runpy.run_path(RAW, run_name="__main__")
except SystemExit as e:
    # Propagate code but avoid long traceback in .bat
    sys.exit(e.code if isinstance(e.code, int) else 1)
except Exception as e:
    print("Ocurrió un error ejecutando el pipeline:", e)
    raise
