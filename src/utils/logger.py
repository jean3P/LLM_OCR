import logging


def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Check if the logger has handlers already to prevent adding duplicates
    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

# Example usage:
# logger = setup_logger('workflow_logger', 'workflow.log')
# logger.info('This is an info message')
