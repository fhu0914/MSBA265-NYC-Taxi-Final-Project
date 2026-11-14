# NYC Taxi Trip Duration Prediction — MSBA 265 P2

**Status:** P2 draft with a working model, documented evaluation, saved artifact, **three optional bonus extensions (FastAPI, Docker, MLOps drift monitoring)**, report, and 5-slide stakeholder presentation. This is an extension of P1, not a new topic.

## Project outputs

- `docs/P2_Draft_Report.docx`: 10-section draft with methods, evaluated findings, visuals, limitations, P1 feedback placeholder.
- `docs/P2_Stakeholder_Draft.pptx`: ~5-minute stakeholder presentation.
- `reports/summary.html`: open in browser for readable results and plots.
- `reports/data_to_action_pipeline.png`, `reports/model_comparison.png`, `reports/errors_by_pickup_hour.png`, `reports/absolute_errors.png`.
- `reports/model_comparison.csv` and `reports/summary.json`: machine-readable evidence.
- `artifacts/taxi_eta_pipeline.joblib`: trained and reloaded model (version-sensitive).
- `src/p2_analysis.py` and `src/check_saved_model.py`: analysis/reproduction.
- `app.py`, `tests/test_api.py`: optional FastAPI bonus demo and tests.
- `Dockerfile`, `.dockerignore`: optional Docker bonus; builds and runs the service with its dependencies.
- `src/make_monitoring_baseline.py`, `src/monitor.py`, `src/check_monitor_sensitivity.py`, `tests/test_monitor.py`: optional MLOps bonus; drift detection with measured alarm thresholds.
- `artifacts/monitoring_baseline.json`: reference distributions the monitor compares live traffic against.
- `P1.pdf`, `src/initial_eda.py`: preserve the P1 materials for continuity.

## Required Kaggle data (not distributed)

The competition dataset is **NOT** uploaded to GitHub or included in this project handoff. Obtain the original `train.zip` yourself from:

https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data

Log in, accept the competition's terms, download only `train.zip`, and place it at **`data/train.zip` without unzipping**. The model artifact contains learned coefficients, not the raw trip rows.

## Reproduce the analysis

From the directory containing this README.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/p2_analysis.py
python src/check_saved_model.py
python -m pytest -q tests/test_api.py
```

### macOS / Linux Terminal

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 src/p2_analysis.py
python3 src/check_saved_model.py
python3 -m pytest -q tests/test_api.py
```

On macOS the command is `python3`, not `python`. Use `python3 -m pytest` rather than a bare `pytest` so the project root stays on the import path.

For peers who only want to test the saved model and API, `train.zip` is not required until they rerun training. They must still obtain the original dataset and rerun the analysis for full independent reproduction.

Expected output of `src/check_saved_model.py`:

```text
PASS: reloaded model prediction 5km, 14:00 -> 12.98 minutes
```

Expected output of the test command:

```text
4 passed
```

A scikit-learn `InconsistentVersionWarning` may appear when the artifact loads: it was fitted with scikit-learn 1.8.0. The prediction is unaffected and still returns 12.98 minutes.

## Bonus: run FastAPI and test it

### Windows PowerShell

```powershell
uvicorn app:app --reload
```

### macOS / Linux Terminal

```bash
python3 -m uvicorn app:app --reload
```

Open **http://127.0.0.1:8000/docs**. Expand **POST /predict → Try it out** and paste:

```json
{"distance_km":5,"pickup_hour":14}
```

Expected: HTTP 200 and

```json
{"predicted_minutes":12.98,"model":"distance + pickup hour linear regression","note":"Illustrative estimate; great-circle distance is not road distance."}
```

Send `"pickup_hour":31` instead and Pydantic range validation rejects it with HTTP 422:

```json
{"detail":[{"type":"less_than_equal","loc":["body","pickup_hour"],"msg":"Input should be less than or equal to 23","input":31,"ctx":{"le":23}}]}
```

`GET /health` returns `{"status":"ok","model_available":true}` and confirms the artifact is present.

A captured run of this endpoint is included at `docs/api_demo.png`.

To verify from a second terminal without the browser:

**macOS / Linux**

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"distance_km":5,"pickup_hour":14}'
```

**Windows PowerShell**

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType "application/json" -Body '{"distance_km":5,"pickup_hour":14}'
```

The four checks in `tests/test_api.py` cover `/health`, a valid prediction, an out-of-range pickup hour, and a negative distance.

## Bonus: Docker

Packages the model and its API so a reviewer can run the prototype without
installing Python or any dependency by hand. The image carries the fitted
artifact only; `.dockerignore` keeps the Kaggle dataset out of it.

Build:

```bash
docker build -t taxi-eta .
```

Run the reproducible test suite inside the image:

```bash
docker run --rm taxi-eta python -m pytest -q tests/
```

Run the service:

```bash
docker run --rm -p 8000:8000 taxi-eta
```

Then open **http://127.0.0.1:8000/docs** and send the same request as above;
`{"distance_km":5,"pickup_hour":14}` returns `predicted_minutes` 12.98, exactly
as it does outside the container.

The commands are identical on Windows PowerShell, macOS and Linux.

## Bonus: MLOps — drift monitoring

A deployed model never sees the real trip duration, so live accuracy cannot be
measured directly. What can be measured is whether the traffic arriving at the
API still resembles the data the model was fitted on, and whether the
predictions it returns still look normal. A model rarely fails loudly; it fails
quietly on inputs it was never trained for.

Every served prediction is appended to `logs/predictions.jsonl`. The monitor
compares that traffic against a reference built from the training rows, using
the **Population Stability Index (PSI)** with the standard bands:

| PSI | Reading | Action |
|---|---|---|
| below 0.10 | no meaningful shift | none |
| 0.10 to 0.25 | moderate shift | keep watching, do not retrain yet |
| above 0.25 | significant shift | investigate before trusting the estimates |

Build the reference, then check the log:

```bash
python src/make_monitoring_baseline.py     # needs data/train.zip; writes artifacts/monitoring_baseline.json
python src/monitor.py                      # writes reports/monitoring_report.json
```

On macOS use `python3`. `src/monitor.py` exits with code 1 when any variable
crosses the investigate threshold, so the check can later be wired into a
scheduled job.

### Why the monitor waits for 500 requests

An alarm is only useful if it stays quiet when nothing is wrong. PSI over ten
bins is noisy on small samples, so the minimum sample size was measured rather
than assumed:

```bash
python src/check_monitor_sensitivity.py    # writes reports/monitor_sensitivity.csv
```

It repeatedly draws traffic that matches the baseline by construction, so every
alarm it records is a false alarm:

| Requests | False "watch" | False "investigate" |
|---:|---:|---:|
| 30 | 97.2% | 68.2% |
| 60 | 75.8% | 14.8% |
| 100 | 40.2% | 0.2% |
| 200 | 2.8% | 0.0% |
| 500 | 0.0% | 0.0% |

At 30 requests a perfectly healthy service would be sent for investigation two
times out of three, which would simply teach the operator to ignore the alarm.
500 is the first round figure where both false-alarm rates reach zero, so that
is the threshold the monitor uses.

### Demonstrated behaviour

Healthy traffic drawn from the training distribution:

```text
distance_km        PSI  0.0064  stable
pickup_hour        PSI  0.0179  stable
predicted_minutes  PSI  0.0222  stable
Verdict: STABLE
```

A simulated ingestion fault sending only long overnight trips:

```text
distance_km        PSI 12.4338  investigate   live mean  66.963 vs baseline 3.433
pickup_hour        PSI 12.5229  investigate   live mean   1.487 vs baseline 13.603
predicted_minutes  PSI 12.4338  investigate   live mean 360.000 vs baseline 13.051
Verdict: INVESTIGATE
```

The second case shows why the monitoring matters: the service returned HTTP 200
for every one of those requests. The log-linear model, extrapolated far beyond
its training range, saturated at the 360-minute output cap instead of failing.
Nothing in the API response would have revealed the problem.

### Limitations

Drift detection answers whether the inputs changed, not whether the model is
still correct; confirming that needs actual trip durations, which arrive only
after each trip ends. The reference is a fixed 2016 training sample, so genuine
seasonal change would also register as drift and has to be ruled out by a human
before retraining.

## Latest actual evaluated results

| Model | RMSLE (↓) | MAE (min ↓) | % within ±3 min (↑) |
|---|---:|---:|---:|
| Median baseline | 0.7743 | 7.422 | 29.43% |
| Distance only | 0.5920 | 6.189 | 43.66% |
| Distance + pickup hour | **0.5907** | **6.182** | **43.93%** |

Data: 1,458,644 records read; 1,456,572 included; 2,072 excluded from scoped analysis for trips over 6 hours or great-circle distances over 200 km. Fixed random 80/20 split with seed 42; 291,315 validation rows. Predictions are clipped to 0–360 minutes, with model trained on `log1p(trip_duration seconds)`. This is a limited proof of concept, **not a real-time ETA service**. The source data are from 2016.

> P1 results differ because P1 used the full sample without this explicit scoped anomaly policy. **Do not claim P2 has improved the model solely by comparing cross-stage scores.** Model comparisons inside the P2 table are on the same sample and thus meaningful.

## Assumptions, limitations & integrity

1. Destination coordinates are known at prediction time; great-circle distance is not travel-route distance. Do not use `dropoff_datetime` (target leakage).
2. Excluding trips beyond 6 hours is an outcome-based proof-of-concept scope; production coverage and fairness require further analysis. GPS anomalies also motivate the distance cap.
3. The 80/20 random holdout is *retrospective*, not a chronological forecast test. Some uncertainty in atypical hours and trips remains.
4. No causal or monetary savings claims. Presentation communicates *candidate* decisions, not proven operational gains.
5. AI was used for code, writing, and debugging support; analysis outputs were run and checked. The student should review and be able to explain decisions.

## What remains before Canvas submission

- Review the drafted report/PPT yourself; check alignment with professor's current P2 due date and Canvas entry.
- Add genuine P1 instructor feedback when available (currently not supplied).
- **Ask a real peer to clone/pull your final GitHub commit, follow this README, verify outputs independently, and sign a NEW Assignment Validation PDF for P2. Do not fabricate the signature.**
- Push this P2 material to your GitHub repository; use `git status` before `git add` and confirm that **data/train.zip is ignored**. Submit repository URL, commit SHA, completed validation form, and P2 deliverables per Canvas directions.

Source: [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/competitions/nyc-taxi-trip-duration). Course specification: teacher-provided `FInal-Project.docx` and `assignment-validation.pdf`.
