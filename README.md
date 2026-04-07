# mobilenet_yolo

This repository is only a test build.

The current version is a macOS desktop prototype for realtime object detection. It uses:

- `PySide6` for the GUI and camera access
- `OpenCV DNN` for inference
- `YOLOX` ONNX for object detection

## Current Status

- Experimental
- Focused on local testing, not production release
- Tuned for working camera access on macOS app bundles
- Still being adjusted for latency, accuracy, and detection stability

## Why `build/`, `dist/`, and `node_modules/` are not in git

Those folders were not uploaded because they are generated artifacts, not source files:

- `build/`: temporary packaging output from PyInstaller
- `dist/`: generated app bundles and executables
- `node_modules/`: old dependency leftovers from the earlier broken Node-based version

They can be rebuilt locally from the tracked source files.

## Main Files

- `src/mobilenet_gui.py`: current application source
- `config/requirements.txt`: Python dependencies
- `config/mobilenet_gui.spec`: PyInstaller packaging config

## Notes

This project started from an old broken MobileNet-on-Node experiment and was rebuilt into a Python GUI prototype with camera-based object detection.
