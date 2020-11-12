from pathlib import Path
from typing import Optional

from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer


ROOT_DIRECTORY = Path("/codeexecution")
RUNTIME_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"

DEFAULT_SUBMISSION_FORMAT = DATA_DIRECTORY / "submission_format.csv"
DEFAULT_OUTPUT = ROOT_DIRECTORY / "submission.csv"


def main(
    submission_format: Path = DEFAULT_SUBMISSION_FORMAT,
    output_file: Optional[Path] = DEFAULT_OUTPUT,
):
    """
    Generate an example submission. The defaults are set so that the script will run successfully
    without being passed any arguments, invoked only as `python main.py`.
    """
    logger.info("loading parameters")
    
    # read in the submission format
    logger.info(f"reading submission format from {submission_format} ...")
    submission_format = pd.read_csv(submission_format, index_col=0)
    logger.info(f"read dataframe with {len(submission_format):,} rows")

    submission = submission_format.copy()
    if output_file is not None:
        logger.info(f"writing {len(submission):,} rows out to {output_file}")
        submission.to_csv(output_file, index=True)

    return submission_format


if __name__ == "__main__":
    typer.run(main)
