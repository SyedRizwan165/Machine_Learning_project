import sys
from src.logger import logging

def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error information, including the script name, line number, and error message.

    Args:
        error (Exception): The error object.
        error_detail (sys): The sys module to fetch detailed exception information.

    Returns:
        str: A formatted string with the error details.
    """
    # Retrieve exception traceback details
    _, _, exc_tb = error_detail.exc_info()
    
    # Get the file name where the exception occurred
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    # Format the error message with the script name, line number, and error message
    error_message = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message


class CustomException(Exception):
    """
    A custom exception class to handle and log errors with detailed information.
    """
    def __init__(self, error_message, error_detail: sys):
        """
        Initializes the CustomException instance with a detailed error message.

        Args:
            error_message (str): The error message to be displayed.
            error_detail (sys): The sys module to fetch detailed exception information.
        """
        # Call the base class constructor with the original error message
        super().__init__(error_message)
        
        # Generate a detailed error message
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        """
        Returns the detailed error message as the string representation of the exception.

        Returns:
            str: The detailed error message.
        """
        return self.error_message
