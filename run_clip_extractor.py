#!/usr/bin/env python3
"""Viral Clip Extractor — entry point"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from viral_clip_extractor.cli import main

if __name__ == "__main__":
    main()
