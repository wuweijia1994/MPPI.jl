import logging
import argparse
import h5py

parser = argparse.ArgumentParser(description='Run the Guided Policy Search algorithm.')
parser.add_argument('name', help='select one of the following name: train, valid, test')
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    #filename='output.log',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s')
logger = logging.getLogger(__name__)

f = h5py.File(args.name+'.hdf5','r+')
f = f["/action"]
for key in f:
    logger.info(key)
    logger.info("shape: {}".format(f[key].shape))
    logger.info("[5]: {}".format(f[key][5]))

