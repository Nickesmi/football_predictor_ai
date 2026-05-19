"""
pyi_rth_xgboost.py — PyInstaller runtime hook for XGBoost

Ensures libxgboost.dylib is discoverable by XGBoost's libpath.find_lib_path()
when running from a frozen PyInstaller bundle.

XGBoost looks for the dylib relative to its own __file__ path.
In a frozen bundle sys._MEIPASS/xgboost/lib/libxgboost.dylib is the target.
"""

import os
import sys
import shutil

if getattr(sys, "frozen", False):
    base = sys._MEIPASS  # type: ignore[attr-defined]

    # XGBoost looks for: <xgboost_package_dir>/lib/libxgboost.dylib
    # In our bundle: <_MEIPASS>/xgboost/lib/libxgboost.dylib
    xgb_lib_src = os.path.join(base, "xgboost", "lib", "libxgboost.dylib")

    # Also check Frameworks (macOS .app structure)
    frameworks = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(base))),
        "Frameworks",
    )
    xgb_lib_alt = os.path.join(frameworks, "xgboost", "lib", "libxgboost.dylib")

    # Inject the correct path into sys.path so xgboost.__file__ resolves correctly
    xgb_package_dir = os.path.join(base, "xgboost")
    if os.path.isdir(xgb_package_dir) and xgb_package_dir not in sys.path:
        sys.path.insert(0, base)

    # Set DYLD_LIBRARY_PATH so the dynamic linker can find libxgboost
    lib_dir = os.path.join(base, "xgboost", "lib")
    if os.path.isdir(lib_dir):
        existing = os.environ.get("DYLD_LIBRARY_PATH", "")
        os.environ["DYLD_LIBRARY_PATH"] = lib_dir + (":" + existing if existing else "")
