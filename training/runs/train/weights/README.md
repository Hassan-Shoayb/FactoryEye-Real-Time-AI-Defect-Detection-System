# Trained Model Weights Directory

Place your fine-tuned model weights in this directory:
- `best.pt`: The model checkpoint with the highest validation mAP50.
- `last.pt`: The model checkpoint from the final training epoch.

### How to place your model:
If you trained your model in Google Colab:
1. Download `best.pt` from Colab.
2. Move it to this directory:
   ```bash
   mv ~/Downloads/best.pt training/runs/train/weights/best.pt
   ```

The FastAPI inference service in `api/` will automatically load `best.pt` from this path on startup.
