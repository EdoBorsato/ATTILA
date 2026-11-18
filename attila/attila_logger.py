import logging

logger = logging.getLogger(__name__)

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log message format
    handlers=[
        logging.FileHandler("example.log"),  # Log messages to a file
        logging.StreamHandler()  # Also log messages to the console
    ]
)

