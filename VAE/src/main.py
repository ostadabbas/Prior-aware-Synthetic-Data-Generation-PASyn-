import os
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.data.prepare_data import prepare_vposer_datasets

expr_code = 1

amass_dir = './data/'

vposer_datadir = makepath('./result/%s' % (expr_code))

logger = log2file(os.path.join(vposer_datadir, '%s.log' % (expr_code)))
logger('[%s] Preparing data for training VPoser.'%expr_code)

amass_splits = {
    'vald': ['vald'],
    'test': ['test'],
    'train': ['train']
}
amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))

prepare_vposer_datasets(vposer_datadir, amass_splits, amass_dir, logger=logger)