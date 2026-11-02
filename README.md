# MSBA 265 Final Project
## New York City Taxi Trip Duration

**P1 Combined Submission PDF:** [P1.pdf](./P1.pdf)

### Project Overview

This project uses the New York City Taxi Trip Duration dataset to study and predict taxi trip duration.

The goal is to build a reproducible analytics workflow that connects a real business problem with data exploration, modeling, evaluation, and actionable interpretation.

### Business Problem

Trip-time uncertainty can affect taxi fleet planning, driver utilization, and passenger expectations. A taxi fleet operations or urban mobility planning team could use more reliable trip-duration estimates to support operational decisions.

### Stakeholder

The primary stakeholder is a hypothetical New York City taxi fleet operations or urban mobility planning team.

### Project Question

Using only information available at or before pickup, how accurately can taxi trip duration be predicted?

### Intended Use

The prediction is intended to support trip-time estimation and operational planning at the time of pickup. It is not intended to replace official navigation or safety-critical systems.

This project assumes that the passenger's destination coordinates are known at prediction time.

### Dataset Overview

This project uses the New York City Taxi Trip Duration dataset from the Kaggle competition.

The data is based on 2016 New York City yellow taxi trip records. The training set contains 1,458,644 trips, and the test set contains 625,134 trips.

Each row represents one taxi trip.
# Converted for the data audit only. Excluded as a predictor: it is unknown
# at pickup time and would leak the target.

Important variables include:

- `vendor_id`
- `pickup_datetime`
- `dropoff_datetime`
- `passenger_count`
- `pickup_longitude`
- `pickup_latitude`
- `dropoff_longitude`
- `dropoff_latitude`
- `store_and_fwd_flag`

The target variable is `trip_duration`, measured in seconds.

`dropoff_datetime` will not be used as a predictor because it is only available after the trip is completed and would directly reveal information about the target.

### Data Setup

This project uses the Kaggle *New York City Taxi Trip Duration* competition data,
which is not included in this repository.

To obtain it:

1. Create a free Kaggle account at https://www.kaggle.com if you do not have one.
2. Open the competition page:
   https://www.kaggle.com/competitions/nyc-taxi-trip-duration
3. Go to the **Rules** tab and click **I Understand and Accept**.
   Downloads stay locked until you do this.
4. Go to the **Data** tab and download **`train.zip`** (about 63 MB).
5. Place it at `data/train.zip` inside this repository.

The file must be named exactly `train.zip` and must contain a single file,
`train.csv`. Do not unzip it; the script reads the zip directly.

### How to Run

Clone the repository:

```bash
git clone https://github.com/fhu0914/MSBA265-NYC-Taxi-Final-Project.git
cd MSBA265-NYC-Taxi-Final-Project
```

### Environment

- Python 3.10 or newer
- Tested on Windows with Python 3.13

Create and activate a virtual environment:

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS Terminal**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Create the data folder:

```bash
mkdir data
```

Download `train.zip` from the Kaggle link above and place it here:

`data/train.zip`

Run the P1 analysis:

```bash
python src/initial_eda.py
```

The script will display several figures. Close each figure window to allow the script to continue running.

A successful run should produce approximately:

- Baseline validation RMSLE: `0.7971`
- Distance-only linear regression RMSLE: `0.6388`
- Distance + pickup hour linear regression RMSLE: `0.6376`

It should also generate the PNG files in the `reports/` folder.

### Proposed Method

I will begin with a data quality audit and exploratory data analysis.

I plan to engineer interpretable features that are available at or before pickup, including:

- estimated trip distance from pickup and dropoff coordinates
- pickup hour
- day of week
- month
- passenger count

I will first establish a simple baseline and then build a linear regression model.

Because the Kaggle competition evaluates predictions using RMSLE, I plan to model trip duration on a log scale and compare model performance using RMSLE.

### Success Measure

The primary evaluation metric will be Root Mean Squared Logarithmic Error (RMSLE).

A successful model should improve on the baseline while remaining interpretable and using only information that would realistically be available at prediction time.

### Initial Feasibility Results

A simple median baseline predicts 662 seconds for every trip and achieved a validation RMSLE of 0.7971.

A linear regression model using only estimated trip distance improved the validation RMSLE to 0.6388. Adding pickup hour produced a small additional improvement, reducing validation RMSLE from 0.6388 to 0.6376.

This initial result suggests that trip distance contains useful predictive information and that the project is feasible for further modeling.

The initial data audit also identified extreme values, including unusually long trip durations and very large calculated trip distances. These observations will be investigated before any filtering decisions are made.

This suggests that trip distance is the strongest predictor tested so far, while pickup time provides smaller incremental value.

### Initial Findings

The target variable, `trip_duration`, is strongly right-skewed. For visualization, the histogram is limited to observations at or below the 99th percentile so that the main distribution is readable; this display cutoff does not remove observations from the modeling data. The model uses `log1p(trip_duration)` because the target is highly right-skewed.

Median trip duration varies substantially by pickup hour, from about 7.9 minutes at 6:00 to about 12.1 minutes at 14:00. However, adding `pickup_hour` to the distance-only linear regression improves validation RMSLE only slightly, from 0.6388 to 0.6376. One likely reason is that hour is currently treated as a simple numeric feature, while the observed daily pattern is nonlinear.

Estimated trip distance appears to be the strongest predictor tested so far. A distance-only linear regression substantially improved validation RMSLE compared with the median baseline, while adding pickup hour produced a smaller additional improvement.

### Next Steps

The next stage of the project will:

- investigate extreme trip durations and unusually large calculated trip distances before making any filtering decisions
- evaluate additional pickup-time features such as day of week and month
- test whether additional available predictors improve validation RMSLE
- compare more complete models with the current median baseline and simple linear regression models
- maintain a leakage-free workflow using only information that would be available at prediction time
- treat pickup hour as a categorical or cyclical feature rather than a linear
  numeric value; its current linear encoding may explain the small improvement

### AI Use Disclosure

AI tools were used to assist with code generation, debugging, documentation, and explanations of technical concepts.

I reviewed, tested, and revised the AI-assisted code, verified the outputs, and used the results to support my own analysis and interpretation.

AI was used as a support tool rather than as a substitute for understanding the project workflow.