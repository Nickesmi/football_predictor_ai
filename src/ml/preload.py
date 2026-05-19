import os
import sys
import ctypes

def preload_xgboost() -> bool:
    """
    Preload libxgboost.dylib on macOS frozen environments to bypass
    PyInstaller sandbox/SIP issues with dynamic library loading.
    """
    try:
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(__file__))

        possible_paths = [
            os.path.join(base_path, "xgboost/lib/libxgboost.dylib"),
            os.path.join(base_path, "lib/libxgboost.dylib"),
            os.path.join(base_path, "Frameworks/lib/libxgboost.dylib"),
            os.path.join(base_path, "Frameworks/xgboost/lib/libxgboost.dylib"),
        ]

        for path in possible_paths:
            if os.path.exists(path):
                ctypes.cdll.LoadLibrary(path)
                return True
    except Exception:
        pass

    return False
