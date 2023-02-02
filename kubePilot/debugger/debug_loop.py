import logging
logging.basicConfig(level=logging.INFO)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)
LOGGER.info("Inside debugging loop...")
while True:
    pass