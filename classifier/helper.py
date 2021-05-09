import logging
import time
import datetime
from pathlib import Path
import re


def init_logger(args, config, path, name=''):
    start_time = time.time()
    timestamp = datetime.datetime.fromtimestamp(start_time).strftime('%Y%m%d-%H%M%S')

    filename = Path(path) / (str(timestamp)+'.log')

    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=filename, level=logging.INFO, format=FORMAT)
    logging.info('Begin logging...')
    logging.info('Start Time:\t'+datetime.datetime.fromtimestamp(start_time).strftime('%H:%M:%S Date: %d, %b %Y'))
    logging.info(f'Current path: {Path().absolute()}')

    logging.info('START Args')
    for arg, value in sorted(vars(args).items()):
        logging.info(f'Arg {arg}: {value}')
    logging.info('END Args')

    logging.info('START Config file')
    for section in config.sections():
        for item in config[section]:
            value = config[section][item]
            logging.info(f'Section: {section}, Key: {item}, Value: {value}')
    logging.info('END Config file')


def make_output_dir(args, config, name='model'):
    if '/' in name:
        name = '_'.join(name.split('/'))
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    output_dir = Path(config['CONFIG']['output_dir']) / '_'.join([name, timestamp])
    if not output_dir.is_dir():
        output_dir.mkdir(parents=True)

    return output_dir


def replace_html(text):
    return re.sub(r'http\S+', '', string)
    