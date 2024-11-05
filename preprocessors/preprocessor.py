import logging


class Preprocessor:
    def __init__(self, name="Preprocessor") -> None:
        # Set up the logger with the given name
        self.logger = logging.getLogger(name)

        # Check if the logger has handlers already (to avoid duplicate handlers)
        if not self.logger.hasHandlers():
            # Configure logging only if there are no handlers yet
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def transform(self, df):
        pass
