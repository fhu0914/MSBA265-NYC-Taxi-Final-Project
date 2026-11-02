# MSBA 265 Final Project
## New York City Taxi Trip Duration

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

This suggests that trip distance is the dominant early predictor, while pickup time may provide smaller incremental value.

### Initial Findings

The target variable, `trip_duration`, is strongly right-skewed in its original scale. After applying a log transformation, the main distribution becomes much easier to interpret.

Median trip duration also varies by pickup hour. Trips are generally shorter in the early morning and longer during the daytime, suggesting that pickup time may be a useful predictor of trip duration.

Estimated trip distance appears to be the strongest initial predictor. A distance-only linear regression substantially improved validation RMSLE compared with the median baseline, while adding pickup hour produced a smaller additional improvement.

### Next Steps

The next stage of the project will:

- investigate extreme trip durations and unusually large calculated trip distances before making any filtering decisions
- evaluate additional pickup-time features such as day of week and month
- test whether additional available predictors improve validation RMSLE
- compare more complete models with the current median baseline and simple linear regression models
- maintain a leakage-free workflow using only information that would be available at prediction time

### AI Use Disclosure

AI tools were used to assist with code generation, debugging, documentation, and explanations of technical concepts.

I reviewed, tested, and revised the AI-assisted code, verified the outputs, and used the results to support my own analysis and interpretation.

AI was used as a support tool rather than as a substitute for understanding the project workflow.