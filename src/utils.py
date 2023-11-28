import datetime
from dataclasses import dataclass
from pathlib import Path
from from_root import from_root

from loguru import logger


@dataclass
class Timediff:
    """Stores time difference"""
    days: int
    hours: int
    minutes: int
    seconds: float


def get_file_age(file_path: Path):
    """Gets age of a file"""

    time_now = datetime.datetime.now()

    try:

        file_time = datetime.datetime.fromtimestamp(file_path.stat().st_mtime)

    except FileNotFoundError as e:

        logger.info(f"{str(file_path)} file is not found")
        
        file_age = None

    else:

        # convert time difference to days, hours, minutes, secs
        td = (time_now - file_time)
        days = td.days
        hours, remainder = divmod(td.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        seconds += td.microseconds / 1e6

        file_age = Timediff(*(days, hours, minutes, seconds))

    return file_age


if __name__ == "__main__":

    from loguru import logger

    ROOT = from_root()
    file_path = ROOT / 'data' / 'master' / 'snp_indexes.yml'

    td = get_file_age(file_path)

    logger.info(td)

