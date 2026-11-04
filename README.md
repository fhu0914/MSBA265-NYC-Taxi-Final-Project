# MSBA 265 Final Project
## New York City Taxi Trip Duration

**P1 Combined Submission PDF:** [P1.pdf](./P1.pdf)

### ⚠️ START HERE: Required Data Before Running

The raw Kaggle dataset is not included in this GitHub repository, and this is required by the competition rules. Section 7.B of the NYC Taxi Trip Duration rules states that participants may "not transmit, duplicate, publish, redistribute or otherwise provide or make available the Data to any party not participating in the Competition." Sharing the analysis code is explicitly permitted under Section 8.B, which is why this repository contains the code but not the data.

Each user must therefore download the data themselves under their own accepted rules. Before running the analysis, you must download **one specific file: `train.zip`** from the official Kaggle competition page.

#### Step 1: Create or sign in to a Kaggle account

If you do not already have a Kaggle account, create a free account at:

https://www.kaggle.com

Then sign in.

#### Step 2: Open the competition page

Go to:

https://www.kaggle.com/competitions/nyc-taxi-trip-duration

#### Step 3: Accept the competition rules

Open the **Rules** tab.

If Kaggle asks you to accept the competition rules, click:

**I Understand and Accept**

You may not be able to download the data until this step is completed.

#### Step 4: Open the Data page

Go to the **Data** tab, or open this link directly:

https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data

On the right side of the page, find the **Data Explorer** panel.

You should see three files:

- `sample_submission.zip`
- `test.zip`
- `train.zip`

Click **`train.zip`**.

#### Step 5: Download only `train.zip`

After selecting `train.zip`, confirm that the main file panel shows:

`train.zip` (about 66 MB)

Then click the small **download icon (down arrow)** in the **upper-right corner of the main file panel**.

**Do not click the large `Download All` button.**

For this project, you do not need:

- `sample_submission.zip`
- `test.zip`
- the KaggleHub Python API

You only need **`train.zip`**.

#### Step 6: Place the file in the project

Keep the downloaded file zipped.

**Do not unzip it.**

Place it in this exact location:

`data/train.zip`

Before running the analysis, your project should contain:

```text
MSBA265-NYC-Taxi-Final-Project/
├── data/
│   └── train.zip
├── reports/
├── src/
│   └── initial_eda.py
├── P1.pdf
├── README.md
└── requirements.txt
```

The script reads train.zip directly.

`train.zip` will exist only in your local copy of the project after you download it.

It is intentionally not stored or displayed in the GitHub repository.

If data/train.zip is missing, src/initial_eda.py will intentionally stop and display an error message.

After confirming that data/train.zip exists, continue to the How to Run section below.

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

This project uses the Kaggle *New York City Taxi Trip Duration* competition dataset.

Please complete the **START HERE: Required Data Before Running** section at the top of this README before running the analysis.

The required file is:

`data/train.zip`

Keep the file zipped. The analysis script reads it directly.

### How to Run

#### 1. Clone the repository

```bash
git clone https://github.com/fhu0914/MSBA265-NYC-Taxi-Final-Project.git
cd MSBA265-NYC-Taxi-Final-Project
```

#### 2. Download the required dataset

Before running the code, make sure you have completed the **START HERE** section above.

You must have this file:

```text
data/train.zip
```

The file should remain zipped.

You can confirm that it is in the correct location before continuing.

#### 3. Create and activate a virtual environment

- Python 3.10 or newer
- Tested on Windows with Python 3.13

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

#### 4. Install the required packages

**Windows**

```powershell
python -m pip install -r requirements.txt
```

**macOS**

```bash
python3 -m pip install -r requirements.txt
```

#### 5. Run the analysis

**Windows**

```powershell
python src/initial_eda.py
```

**macOS**

```bash
python3 src/initial_eda.py
```

The script will display several figures. Close each figure window to allow the script to continue running.

A successful run should produce approximately:

- Baseline validation RMSLE: `0.7971`
- Distance-only linear regression RMSLE: `0.6388`
- Distance + pickup hour linear regression RMSLE: `0.6376`

It should also generate:

- `reports/trip_duration_distribution_99pct.png`
- `reports/median_trip_duration_by_hour.png`

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