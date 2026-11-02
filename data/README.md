# Data

This project uses the Kaggle *New York City Taxi Trip Duration* competition
dataset. The raw data is **not** included in this repository: the competition
rules restrict redistribution, and `train.zip` is about 63 MB.

## Source

https://www.kaggle.com/competitions/nyc-taxi-trip-duration

The data originates from the NYC Taxi and Limousine Commission (TLC) 2016
yellow taxi trip records, sampled and cleaned by Kaggle for this competition.

## How to obtain it

See **Data Setup** in the project README. A free Kaggle account is required,
and the competition rules must be accepted before the download unlocks.

## What to expect

- `train.zip` (~63 MB), containing a single file `train.csv`
- 1,458,644 rows x 11 columns
- One row = one completed taxi trip
- Target: `trip_duration`, in seconds

## Access and licensing notes

Use is governed by the Kaggle competition rules. The data contains no
passenger identifiers, but pickup and dropoff coordinates are precise enough
to identify locations, so it is treated as sensitive and kept out of version
control.