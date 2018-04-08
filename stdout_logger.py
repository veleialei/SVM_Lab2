from logger import logger

# logger.log(0, 'Starting logger (type = ' + logger_type + ') at log level ' + str(log_level) + '.')
# logger.log(1, 'Important message.')
# logger.log(2, 'Less important message.')
# logger.log(3, 'Not important message.')
# logger.log(0, 'Ending logger.')

class stdout_logger(logger):
    def log(self, log_level, message):
        print (str(log_level) + ": " + message)
