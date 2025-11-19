# Bio-Vision Phase-1 Web Demo

This repository contains a minimal end-to-end demo for the **Bio-Vision – Analyzer**:

- **Backend**: Python + FastAPI + PyTorch
- **Frontend**: Plain HTML/CSS/JavaScript (no framework)
- **Task**: Single-image medical (skin / wound) analysis using a 9-class MobileNetV3 model

You can attach your trained model checkpoint (for example `mobilenetv3_phase1_best.pt`) under:

```bash
backend/models/mobilenetv3_phase1_best.pt
```

Then run the API and open the web UI in a browser.

---

## Folder structure

```text
biovision_web_demo/
  backend/
    app.py
    requirements.txt
    models/
      mobilenetv3_phase1_best.pt   # <-- put your model here
    static/
      heatmaps/                    # generated heatmaps (optional)
  frontend/
    index.html
    styles.css
    app.js
```

---

## How to run (development)

1. Create and activate a virtual environment (optional but recommended):

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy your trained model checkpoint into `backend/models/` and rename it:

```bash
cp /path/to/your/mobilenetv3_phase1_best.pt backend/models/
```

4. Start the FastAPI server with Uvicorn:

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

5. Open the web UI

You can open the `frontend/index.html` file directly in your browser, or serve it with a simple static server, for example:

```bash
cd frontend
python -m http.server 8080
```

Then open: `http://localhost:8080`

> The frontend expects the backend to be available at `http://localhost:8000`. If you deploy to a different URL, update `API_BASE_URL` in `frontend/app.js`.

---

## API endpoints

### `POST /infer`

* Request: `multipart/form-data` with `image` (PNG/JPG, ≤ 5MB)
* Response (JSON):

```jsonc
{
  "version": "1.0",
  "inference_id": "uuid-123",
  "pred": {
    "label": "08_severe_infection_pus",
    "prob": 0.91,
    "topk": [
      {"label": "08_severe_infection_pus", "prob": 0.91},
      {"label": "07_dfu", "prob": 0.06},
      {"label": "06_ulcer_general", "prob": 0.03}
    ],
    "metrics": {"redness": 0.68, "cyanosis": 0.10, "area_cm2": 3.7},
    "uncertainty": 0.14
  },
  "explain": {
    "heatmap_uri": "/static/heatmaps/uuid-123_heatmap.png"
  },
  "timing_ms": 420
}
```

* The metrics and heatmap are **simple placeholders** based on basic color statistics and a red overlay. You can replace them with real medical metrics and Grad-CAM.

### `POST /cleanup`

* Request JSON:

```json
{ "inference_id": "uuid-123" }
```

* Action: delete temporary files related to that inference (if any)
* Response:

```json
{ "ok": true }
```

> The frontend calls `/cleanup` whenever a **new image** is selected, to honor the “single-image only” policy.

---

## Classes (Phase-1 label map)

The demo assumes a 9-class model with the following labels:

1. `01_normal`
2. `02_irritation_rash`
3. `03_erythema`
4. `04_dry_cracks`
5. `05_mild_infection`
6. `06_ulcer_general`
7. `07_dfu`
8. `08_severe_infection_pus`
9. `09_burns`

If your model uses a different class order, update `CLASS_NAMES` in `backend/app.py` accordingly.

---

## Notes

* All UI text and code comments are in **English**.
* The UI is designed for a **single image at a time**. When a new image is selected, the previous preview and results are cleared and `/cleanup` is called for the previous `inference_id`.
* This is a **demo**, not a medical device. Do **not** use it for real clinical decisions without proper validation and regulatory approval.
