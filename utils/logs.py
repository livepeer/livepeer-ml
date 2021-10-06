import logging

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s.%(msecs)03d]: %(process)d %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.FileHandler('./logfile.log'), logging.StreamHandler()])

logger = logging.getLogger()