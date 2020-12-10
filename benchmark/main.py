from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger
import pandas as pd
import typer
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

from predict import predict_dst

# default paths for Docker environment
ROOT_DIRECTORY = Path("/codeexecution")
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
DEFAULT_OUTPUT = ROOT_DIRECTORY / "submission.csv"

# constants
INDEX_COLS = ["period", "timedelta"]
MAX_ALLOWED_PREDICTION_TIME_SEC = 30


def get_submission_format(dst_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reshape the data for a t0 and t1 prediction at each hour, but also cut off the first
    seven days of each period to allow some data before making a prediction, and also
    cut off the last timedelta in each period since we don't have a true t1 after the last row.
    """
    # we start each period after one week of data
    t0 = pd.to_timedelta("7 days")
    submission_format = dst_df.loc[dst_df.timedelta >= t0, INDEX_COLS].copy()
    submission_format["t0"] = 0.0
    submission_format["t1"] = 0.0
    # remove the very last observation in each period (since we have no ground truth for that plus one hour)
    for period in submission_format.period.unique():
        period_mask = submission_format.period == period
        t_max = submission_format.loc[period_mask].timedelta.max()
        mask = period_mask & (submission_format.timedelta == t_max)
        submission_format = submission_format.loc[~mask]
    return submission_format.set_index(INDEX_COLS)


def get_ground_truth(
    dst_df: pd.DataFrame, submission_format: pd.DataFrame
) -> pd.DataFrame:
    """
    Given the raw data and the submission format, fill in the zeros with the actual t0 and t1
    values.
    """
    # create a lookup to join t0
    dst0 = dst_df.set_index(INDEX_COLS).dst.rename("t0")
    # create a lookup to join t1 but set the timedelta back an hour so it joins as the _next_ hour
    dst1 = (
        dst_df.assign(timedelta=lambda x: x.timedelta - pd.to_timedelta("1 hour"))
        .set_index(INDEX_COLS)
        .dst.rename("t1")
    )
    # join the two together
    ground_truth = pd.DataFrame(dst0[submission_format.index]).join(dst1)
    return ground_truth


def main(
    data_directory: Path = DATA_DIRECTORY,
    output_file: Optional[Path] = DEFAULT_OUTPUT,
    print_score: bool = False,
):
    """
    Generate an example submission. The defaults are set so that the script will run successfully
    without being passed any arguments, invoked only as `python main.py`.
    """
    _parse_timedelta_col = lambda df: df.assign(timedelta=pd.to_timedelta(df.timedelta))

    dst_path = data_directory / "dst_labels.csv"
    solar_wind_path = data_directory / "solar_wind.csv"
    satellite_positions_path = data_directory / "satellite_positions.csv"
    sunspots_path = data_directory / "sunspots.csv"
    for path in (dst_path, solar_wind_path, satellite_positions_path, sunspots_path):
        assert path.exists(), f"File {path} not found!"

    # read in the raw Dst data
    logger.info(f"reading raw Dst data from {dst_path} ...")
    dst_df = _parse_timedelta_col(pd.read_csv(dst_path))
    logger.info(f"... read dataframe with {len(dst_df):,} rows")

    # calculate a ground truth dataframe
    logger.info(f"calculating submission format and ground truth ...")
    submission_format = get_submission_format(dst_df)
    ground_truth = get_ground_truth(dst_df, submission_format)
    assert (
        submission_format.index == ground_truth.index
    ).all(), "Ground truth index and submission format do not match!"
    del dst_df
    logger.info(
        f"calculated submission format and ground truth dataframes with {len(submission_format):,} rows"
    )

    # read in the dataframes we will be feeding to the prediction function
    logger.info(f"reading in {solar_wind_path} ...")
    solar_wind_df = _parse_timedelta_col(pd.read_csv(solar_wind_path)).set_index(
        INDEX_COLS
    )
    logger.info(f"... read dataframe with {len(solar_wind_df):,} rows")
    logger.info(f"reading in {satellite_positions_path} ...")
    satellite_positions_df = _parse_timedelta_col(
        pd.read_csv(satellite_positions_path)
    ).set_index(INDEX_COLS)
    logger.info(f"... read dataframe with {len(satellite_positions_df):,} rows")
    logger.info(f"reading in {sunspots_path} ...")
    sunspots_df = _parse_timedelta_col(pd.read_csv(sunspots_path)).set_index(INDEX_COLS)
    logger.info(f"... read dataframe with {len(sunspots_df):,} rows")

    submission = submission_format.copy()

    one_minute = pd.to_timedelta("1 minute")
    seven_days = pd.to_timedelta("7 days")
    for period in submission_format.index.get_level_values(0).unique():
        logger.info(f"making predictions for period {period}")
        sub_df = solar_wind_df.loc[period]
        for t0 in tqdm(submission_format.loc[period].index):
            t_minus_7 = t0 - seven_days
            # last seven days of solar wind data except for the current minute
            solar_wind_7d = sub_df[t_minus_7 : t0 - one_minute]
            # last seven satellite positions including for t0 (right now)
            satellite_positions_7d = (
                satellite_positions_df.loc[period].loc[:t0].iloc[-7:, :]
            )
            # last sunspot reading
            latest_sunspot_ssn = sunspots_df.loc[period].loc[:t0].smoothed_ssn[-1]

            # make a prediction for the current timedelta
            start_time = datetime.now()
            dst0, dst1 = predict_dst(
                solar_wind_7d, satellite_positions_7d, latest_sunspot_ssn
            )
            end_time = datetime.now()

            prediction_time_seconds = (end_time - start_time).seconds
            if prediction_time_seconds > MAX_ALLOWED_PREDICTION_TIME_SEC:
                raise RuntimeError(
                    f"Prediction took too long (actual={prediction_time_seconds:0.1f}s, "
                    f"max={MAX_ALLOWED_PREDICTION_TIME_SEC}s) -- exiting!"
                )
            submission.loc[(period, t0), :] = (dst0, dst1)

    if output_file is not None:
        logger.success(f"writing {len(submission):,} rows out to {output_file}")
        submission.to_csv(output_file, index=True)

    if print_score:
        score = mean_squared_error(ground_truth, submission, squared=False)
        logger.info("-" * 80)
        logger.info(f"RMSE: {score:0.2f}")
        logger.info("-" * 80)

    return submission_format


if __name__ == "__main__":
    typer.run(main)
