# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#
# If you use this code in a research publication please consider citing the following:
#
# Expressive Body Capture: 3D Hands, Face, and Body from a Single Image <https://arxiv.org/abs/1904.05866>
#
#
# Code Developed by:
# Nima Ghorbani <https://nghorbani.github.io/>
#
# 2020.12.12

# from pytorch_lightning import Trainer

import glob
import os
import os.path as osp
from datetime import datetime as dt
from pytorch_lightning.plugins import DDPPlugin

import numpy as np
import pytorch_lightning as pl
import torch
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.data.dataloader import VPoserDS
from human_body_prior.data.prepare_data import dataset_exists
from human_body_prior.data.prepare_data import prepare_vposer_datasets
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.tools.angle_continuous_repres import geodesic_loss_R
from human_body_prior.tools.configurations import load_config, dump_config
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import get_support_data_dir
from human_body_prior.tools.omni_tools import log2file
from human_body_prior.tools.omni_tools import make_deterministic
from human_body_prior.tools.omni_tools import makepath
from human_body_prior.tools.rotation_tools import aa2matrot
from human_body_prior.visualizations.training_visualization import vposer_trainer_renderer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import optim as optim_module
from torch.optim import lr_scheduler as lr_sched_module
from torch.utils.data import DataLoader


class VPoserTrainer(LightningModule):
    """

    It includes all data loading and train / val logic., and it is used for both training and testing models.
    """

    def __init__(self, _config):
        super(VPoserTrainer, self).__init__()

        # _support_data_dir = get_support_data_dir()

        vp_ps = load_config(**_config)

        make_deterministic(vp_ps.general.rnd_seed)

        self.expr_id = vp_ps.general.expr_id
        self.dataset_id = vp_ps.general.dataset_id

        self.work_dir = vp_ps.logging.work_dir = makepath(vp_ps.general.work_basedir, self.expr_id)
        self.dataset_dir = vp_ps.logging.dataset_dir = osp.join(vp_ps.general.dataset_basedir, vp_ps.general.dataset_id)

        self._log_prefix = '[{}]'.format(self.expr_id)
        self.text_logger = log2file(prefix=self._log_prefix)

        self.seq_len = vp_ps.data_parms.num_timeseq_frames

        self.vp_model = VPoser(vp_ps)

        # with torch.no_grad():
        #     self.bm_train = BodyModel(vp_ps.body_model.bm_fname)

        self.example_input_array = {'pose_body':torch.ones(vp_ps.train_parms.batch_size, 36),}
        self.vp_ps = vp_ps

    def forward(self, pose_body):

        return self.vp_model(pose_body)

    def _get_data(self, split_name):

        assert split_name in ('train', 'vald', 'test')

        split_name = split_name.replace('vald', 'vald')
        print(split_name)
        assert dataset_exists(self.dataset_dir), FileNotFoundError('Dataset does not exist dataset_dir = {}'.format(self.dataset_dir))
        dataset = VPoserDS(osp.join(self.dataset_dir, split_name), data_fields = ['pose_body'])
        print(len(dataset))
        assert len(dataset) != 0, ValueError('Dataset has nothing in it!')

        return DataLoader(dataset,
                          batch_size=self.vp_ps.train_parms.batch_size,
                          shuffle=True if split_name == 'train' else False,
                          num_workers=self.vp_ps.data_parms.num_workers,
                          pin_memory=True)

    @rank_zero_only
    def on_train_start(self):
        if self.global_rank != 0: return
        self.train_starttime = dt.now().replace(microsecond=0)

        ######## make a backup of vposer
        git_repo_dir = os.path.abspath(__file__).split('/')
        git_repo_dir = '/'.join(git_repo_dir[:git_repo_dir.index('human_body_prior') + 1])
        starttime = dt.strftime(self.train_starttime, '%Y_%m_%d_%H_%M_%S')
        archive_path = makepath(self.work_dir, 'code', 'vposer_{}.tar.gz'.format(starttime), isfile=True)
        cmd = 'cd %s && git ls-files -z | xargs -0 tar -czf %s' % (git_repo_dir, archive_path)
        os.system(cmd)
        ########
        self.text_logger('Created a git archive backup at {}'.format(archive_path))
        dump_config(self.vp_ps, osp.join(self.work_dir, '{}.yaml'.format(self.expr_id)))

    def train_dataloader(self):
        return self._get_data('train')

    def val_dataloader(self):
        return self._get_data('vald')

    def configure_optimizers(self):
        params_count = lambda params: sum(p.numel() for p in params if p.requires_grad)

        gen_params = [a[1] for a in self.vp_model.named_parameters() if a[1].requires_grad]
        gen_optimizer_class = getattr(optim_module, self.vp_ps.train_parms.gen_optimizer.type)
        gen_optimizer = gen_optimizer_class(gen_params, **self.vp_ps.train_parms.gen_optimizer.args)

        self.text_logger('Total Trainable Parameters Count in vp_model is %2.2f M.' % (params_count(gen_params) * 1e-6))

        lr_sched_class = getattr(lr_sched_module, self.vp_ps.train_parms.lr_scheduler.type)

        gen_lr_scheduler = lr_sched_class(gen_optimizer, **self.vp_ps.train_parms.lr_scheduler.args)
        #schedulers = None
        # schedulers = torch.optim.lr_scheduler.ReduceLROnPlateau(gen_optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        schedulers = [
            {
                'scheduler': gen_lr_scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            },
        ]

        return [gen_optimizer], schedulers

    def _compute_loss(self, dorig, drec):
        # l1_loss = torch.nn.L1Loss(reduction='mean')
        smoothl1loss = torch.nn.SmoothL1Loss()
        MSE = torch.nn.MSELoss()

        bs, latentD = drec['poZ_body_mean'].shape
        device = drec['poZ_body_mean'].device

        loss_kl_wt = self.vp_ps.train_parms.loss_weights.loss_kl_wt

        loss_matrot_wt = self.vp_ps.train_parms.loss_weights.loss_matrot_wt

        q_z = drec['q_z']


        # KL loss
        p_z = torch.distributions.normal.Normal(
            loc=torch.zeros((bs, latentD), device=device, requires_grad=False),
            scale=torch.ones((bs, latentD), device=device, requires_grad=False))
        weighted_loss_dict = {
            'loss_kl':loss_kl_wt * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(q_z, p_z), dim=[1])),
        }


        weighted_loss_dict['matrot'] = loss_matrot_wt * smoothl1loss(drec['pose_body'].view(-1, 3),dorig['pose_body'].view(-1, 3))

        weighted_loss_dict['loss_total'] = torch.stack(list(weighted_loss_dict.values())).sum()

        with torch.no_grad():
            unweighted_loss_dict = {'v2v': weighted_loss_dict['loss_total']}
            print('v2v ouput ')
            unweighted_loss_dict['loss_total'] = torch.cat(
                list({k: v.view(-1) for k, v in unweighted_loss_dict.items()}.values()), dim=-1).sum().view(1)

        return {'weighted_loss': weighted_loss_dict, 'unweighted_loss': unweighted_loss_dict}

    def training_step(self, batch, batch_idx, optimizer_idx=None):

        drec = self(batch['pose_body'].view(-1, 36))
        print('drec:',drec['pose_body'][0])
        print('original:', batch['pose_body'][0].view(-1,3))
        loss = self._compute_loss(batch, drec)

        train_loss = loss['weighted_loss']['loss_total']
        print('train_loss', train_loss)
        tensorboard_logs = {'train_loss': train_loss}
        progress_bar = {k: c2c(v) for k, v in loss['weighted_loss'].items()}
        self.log('train_loss', train_loss)
        return {'loss': train_loss, 'progress_bar':progress_bar,  'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        print('val_batch:', batch['pose_body'][0])
        drec = self(batch['pose_body'].view(-1, 36))

        loss = self._compute_loss(batch, drec)
        val_loss = loss['unweighted_loss']['loss_total']
        print('val_loss', val_loss)
        print('rec_loss', loss['weighted_loss']['matrot'])
        progress_bar = {'v2v': val_loss}
        self.log('val_loss', val_loss)
        return {'val_loss': c2c(val_loss)}#, 'progress_bar': progress_bar, 'log': progress_bar}

    def validation_epoch_end(self, outputs):
        metrics = {'val_loss': np.nanmean(np.concatenate([v['val_loss'] for v in outputs])) }
        if self.global_rank == 0:

            self.text_logger('Epoch {}: {}'.format(self.current_epoch, ', '.join('{}:{:.2f}'.format(k, v) for k, v in metrics.items())))
            self.text_logger('lr is {}'.format([pg['lr'] for opt in self.trainer.optimizers for pg in opt.param_groups]))
        metrics = {k: torch.as_tensor(v) for k, v in metrics.items()}
        progress_bar = {'val_loss': metrics['val_loss']}
        return {'val_loss': metrics['val_loss'], 'log': metrics}


    @rank_zero_only
    def on_train_end(self):

        self.train_endtime = dt.now().replace(microsecond=0)
        endtime = dt.strftime(self.train_endtime, '%Y_%m_%d_%H_%M_%S')
        elapsedtime = self.train_endtime - self.train_starttime
        self.vp_ps.logging.best_model_fname = self.trainer.checkpoint_callback.best_model_path

        self.text_logger('Epoch {} - Finished training at {} after {}'.format(self.current_epoch, endtime, elapsedtime))
        self.text_logger('best_model_fname: {}'.format(self.vp_ps.logging.best_model_fname))

        dump_config(self.vp_ps, osp.join(self.work_dir, '{}_{}.yaml'.format(self.expr_id, self.dataset_id)))
        #self.hparams = {}#self.vp_ps.toDict()

    @rank_zero_only
    def prepare_data(self):
        '''' Similar to standard AMASS dataset preparation pipeline:
        Donwload npz file, corresponding to body data from https://amass.is.tue.mpg.de/ and place them under amass_dir
        '''
        self.text_logger = log2file(makepath(self.work_dir, '{}.log'.format(self.expr_id), isfile=True), prefix=self._log_prefix)

        prepare_vposer_datasets(self.dataset_dir, self.vp_ps.data_parms.amass_splits, self.vp_ps.data_parms.amass_dir, logger=self.text_logger)


def create_expr_message(ps):
    expr_msg = '[{}] batch_size = {}.'.format(ps.general.expr_id, ps.train_parms.batch_size)

    return expr_msg


def train_vposer_once(_config):

    resume_training_if_possible = True

    model = VPoserTrainer(_config)
    model.vp_ps.logging.expr_msg = create_expr_message(model.vp_ps)
    # model.text_logger(model.vp_ps.logging.expr_msg.replace(". ", '.\n'))
    dump_config(model.vp_ps, osp.join(model.work_dir, '{}.yaml'.format(model.expr_id)))

    logger = TensorBoardLogger(model.work_dir, name='tensorboard')
    lr_monitor = LearningRateMonitor()

    snapshots_dir = osp.join(model.work_dir, 'snapshots')
    checkpoint_callback = ModelCheckpoint(
        dirpath=makepath(snapshots_dir, isfile=True),
        filename="%s_{epoch:02d}_{val_loss:.2f}" % model.expr_id,
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min',
    )

    #early_stop_callback = EarlyStopping(**model.vp_ps.train_parms.early_stopping)
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=50,verbose=True,mode='min')
    #print('train_params:', model.vp_ps.train_parms.early_stopping)
    resume_from_checkpoint = None
    if resume_training_if_possible:
        available_ckpts = sorted(glob.glob(osp.join(snapshots_dir, '*.ckpt')), key=os.path.getmtime)
        if len(available_ckpts)>0:
            resume_from_checkpoint = available_ckpts[-1]
            model.text_logger('Resuming the training from {}'.format(resume_from_checkpoint))

    trainer = pl.Trainer(gpus=1,
                         weights_summary='top',
                         #distributed_backend='ddp',
                         # replace_sampler_ddp=False,
                         # accumulate_grad_batches=4,
                         # profiler=False,
                         # overfit_batches=0.05,
                         # fast_dev_run = True,
                         # limit_train_batches=0.02,
                         # limit_val_batches=0.02,
                         # num_sanity_val_steps=2,
                         plugins=[DDPPlugin(find_unused_parameters=False)],

                         callbacks=[lr_monitor, early_stop_callback, checkpoint_callback],

                         max_epochs=model.vp_ps.train_parms.num_epochs,
                         logger=logger,
                         resume_from_checkpoint=resume_from_checkpoint
                         )

    trainer.fit(model)
