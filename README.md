# Memento Mori — Lightweight Ethical AI for Art

Memento Mori is an art installation exploring the environmental impact of generative AI by using old-fashioned, lightweight, local, and ethical AI techniques. The project intentionally avoids heavy cloud models and large-scale training runs — instead it favors small local models, frame-based processing, and creative, low-power approaches to generative and interactive art.

## Goals

- Investigate the environmental costs of modern generative AI by contrast: favor local, low-resource methods.
- Create an interactive art installation that runs on modest hardware.
- Demonstrate how creativity can be driven by simple, interpretable pipelines rather than large opaque models.

## Key Concepts

- Lightweight local AI: use compact models and CPU-friendly processing where possible.
- Ethical awareness: design choices and documentation emphasize energy use, reproducibility, and minimal compute.
- Visual/temporal exploration: many scripts operate on frames or live camera input to produce evolving visual output.

## Quick Start

Prerequisites:

- Python 3.7+ (this repo includes a `bin/` virtualenv for reference)
- Common imaging packages (Pillow, OpenCV)

To run a simple script locally:

```bash
python MM_live.py
```

Replace `MM_live.py` with other runners in the repo such as `MM_ACMI.py`, `MM_live_clean.py`, or `MM_live_pixel_decay.py` depending on the visual behavior you want to explore.

## Dependencies

Suggested Python packages (create a virtualenv and install these):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- `Pillow` — image processing and compositing
- `opencv-python` — camera capture and frame operations
- `numpy` — numeric routines used across scripts
- `torch` — optional, only if you want to run model inference locally (CPU or GPU)
- `ultralytics` or `yolov5` — optional, for object-detection helpers if used with `yolov8n.pt`

If you prefer minimal installs, run only `Pillow`, `opencv-python`, and `numpy` to get most local visual behaviors.

## Usage Examples

Run a live CPU-only demo (recommended for galleries):

```bash
python MM_live.py
```

Run the ACMI installation script (adjust camera/index and config inside the file):

```bash
python MM_ACMI.py
```

Run a script that uses object-detection (requires additional model deps):

```bash
pip install torch torchvision ultralytics
python yoloBoxestoPNG.py
```

## Files of Interest

- `Memento.py` / `MementoHF.py` — core experiment scripts
- `MM_live.py` — live demo / real-time processing
- `MM_ACMI.py` — installation-specific script (this workspace's active file)
- `mask.py`, `mask2.py` — masking and compositing helpers
- `yoloBoxestoPNG.py` and `yolov8n.pt` — optional object-detection utilities (if you want region-based processing)

## Design & Ethics Notes

This project is intentionally minimal: rather than training or running large generative models, it focuses on small models, frame-based transformations, heuristic visual mappings, and locally run inference. When heavy models are required, they are optional and clearly labeled — the default experience is low-power and reproducible.

When presenting the work, include the estimated compute and energy footprints for each mode (e.g., live CPU-only, GPU-accelerated, or using `yolov8n.pt`), and prefer the CPU/local modes for gallery demonstrations.

## Contributing

Contributions are welcome. Please open an issue or PR describing the change and its energy/compute implications. For additions that increase resource use, document the trade-offs and provide a low-power fallback where feasible.

## License

This project is licensed under the Creative Commons Attribution-NoDerivatives 4.0 International (CC BY-ND 4.0). That means you are free to copy and redistribute the material in any medium or format for any purpose, including commercial uses, as long as you give appropriate credit and do not distribute modified versions.

See the full `LICENSE` file for details.

## Contact / Presentation

For exhibition or academic contexts, present both the aesthetic output and the ethical rationale: show what was done, what compute was used, and why the low-power approach matters.
