import logging
logging.basicConfig(level=logging.WARNING)
LOG_FORMAT = ('%(levelname) -10s %(asctime)s %(name) -30s %(funcName) '
              '-35s %(lineno) -5d: %(message)s')
LOGGER = logging.getLogger(__name__)

def msg_handler(method_frame, header_frame, body):
    ret_dict = None
    if method_frame:
        if header_frame.content_type == 'init':
            LOGGER.info("Sending init job")
            ret_dict = {'type': 'addQ', 'data': body}
        if header_frame.content_type == 'update_model':
            LOGGER.info("updating model")
            ret_dict = {'type': 'update_model', 'data': body}
        if header_frame.content_type == 'update_config':
            LOGGER.info("updating config")
            ret_dict = {'type': 'update_config', 'data': body}
    return ret_dict