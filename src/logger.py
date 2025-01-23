import logging
import os
from datetime import datetime

# Generate a log file name using the current date and time
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the directory path where logs will be stored
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# Create the directory for logs if it doesn't already exist
os.makedirs(logs_path, exist_ok=True)

# Define the complete path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,  # Specify the file where logs will be written
    format="[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",  # Log message format
    level=logging.INFO,  # Set the logging level to INFO
)
