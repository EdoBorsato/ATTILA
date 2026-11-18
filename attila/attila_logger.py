"""
Package-wide logger of ATTILA.

Importing this module configures the root logger once, through
`logging.basicConfig`, and exposes `logger`, which the pipeline stages
and the `Sources` container use to report their progress.

Every message of level `INFO` or above is written both to the console
and to the file `example.log`, which is created in the current working
directory at import time.
"""
import logging

logger = logging.getLogger(__name__)

# Configure the root logger. This runs at import time, so the log file
# is opened in the directory the importing script is run from.
logging.basicConfig(
    level=logging.INFO,  # Discard messages below the INFO level.
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Time, logger name, level, text.
    handlers=[
        logging.FileHandler("example.log"),  # Append to a file.
        logging.StreamHandler()  # Also print to the console (stderr).
    ]
)

