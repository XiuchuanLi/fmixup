import logging
import sys

import coloredlogs
import verboselogs


def get_logger(logger_name, level="SPAM"):
    verboselogs.install()
    logger = logging.getLogger(logger_name)

    field_styles = {
        "hostname": {"color": "magenta"},
        "programname": {"color": "cyan"},
        "name": {"color": "blue"},
        "levelname": {"color": "black", "bold": True, "bright": True},
        "asctime": {"color": "green"},
    }
    coloredlogs.install(
        level=level,
        logger=logger,
        fmt="%(asctime)s,%(msecs)03d %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s",
        field_styles=field_styles,
    )

    return logger


def display_progress_bar(cur, total):
    bar_len = 30
    filled_len = int(cur / total * bar_len)
    bar_waiter = "=" * filled_len + "." * (bar_len - filled_len)
    sys.stdout.write(f"\r{cur}/{total} [{bar_waiter}] ")
    sys.stdout.flush()
