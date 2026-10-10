#!/usr/bin/env python3
"""Export static brand assets to docs/brand. usage: python scripts/export_brand.py"""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from brand import export_static
export_static(sys.argv[1] if len(sys.argv) > 1 else "docs/brand")
print("ok")
