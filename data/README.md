# Data

The raw dataset is intentionally not stored in this repository.

To run the project, download **`train.zip`** from the official Kaggle competition page:

https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data

On the Kaggle **Data** page:

1. Look at the **Data Explorer panel on the right side**.
2. Select **`train.zip`**.
3. Confirm that the main file panel shows `train.zip` (about 66 MB).
4. Click the small **download icon (down arrow)** in the upper-right corner of the file panel.
5. Do **not** use `sample_submission.zip`, `test.zip`, or the large `Download All` button.
6. Keep the file zipped.
7. Place it in this folder so the final path is:

`data/train.zip`

Then return to the main project README and follow the **How to Run** instructions.

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