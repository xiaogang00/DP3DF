import logging
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DataParallel, DistributedDataParallel
import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.loss import CharbonnierLoss

logger = logging.getLogger('base')


class ltv_loss(nn.Module):
    def __init__(self, alpha=1.2, beta=1.5, eps=1e-4):
        super(ltv_loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps=eps

    def forward(self, content, origin, weight):
        I = origin[:, 0:1, :, :] * 0.299 + origin[:, 1:2, :, :] * 0.587 + origin[:, 2:3, :, :] * 0.114
        L = torch.log(I + self.eps)
        dx = L[:, :, :-1, :-1] - L[:, :, :-1, 1:]
        dy = L[:, :, :-1, :-1] - L[:, :, 1:, :-1]
        dx = self.beta / (torch.pow(torch.abs(dx), self.alpha) + self.eps)
        dy = self.beta / (torch.pow(torch.abs(dy), self.alpha) + self.eps)

        x_loss = dx * ((content[:, :, :-1, :-1] - content[:, :, :-1, 1:]) ** 2)
        y_loss = dy * ((content[:, :, :-1, :-1] - content[:, :, 1:, :-1]) ** 2)
        tvloss = torch.sum(x_loss + y_loss) / 2.0
        return tvloss * weight


class VideoBaseModel(BaseModel):
    def __init__(self, opt):
        super(VideoBaseModel, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training
        train_opt = opt['train']

        # define network and load pretrained models
        self.netG = networks.define_G(opt).to(self.device)

        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        if self.is_train:
            self.netG.train()

            #### loss
            loss_type = train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.cri_pix = nn.L1Loss(reduction='sum').to(self.device)
            elif loss_type == 'l2':
                self.cri_pix = nn.MSELoss(reduction='sum').to(self.device)
            elif loss_type == 'cb':
                self.cri_pix = CharbonnierLoss().to(self.device)
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))
            self.l_pix_w = train_opt['pixel_weight']

            self.cri_pix_ill = nn.L1Loss(reduction='sum').to(self.device)
            self.cri_ltv = ltv_loss()

            #### optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            if train_opt['ft_tsa_only']:
                normal_params = []
                tsa_fusion_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        if 'tsa_fusion' in k:
                            tsa_fusion_params.append(v)
                        else:
                            normal_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))
                optim_params = [
                    {  # add normal params first
                        'params': normal_params,
                        'lr': train_opt['lr_G']
                    },
                    {
                        'params': tsa_fusion_params,
                        'lr': train_opt['lr_G']
                    },
                ]
            else:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    if v.requires_grad:
                        optim_params.append(v)
                    else:
                        if self.rank <= 0:
                            logger.warning('Params [{:s}] will not optimize.'.format(k))

            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            #### schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError()

            self.log_dict = OrderedDict()

    def feed_data(self, data, need_GT=True):
        self.var_L = data['LQs'].to(self.device)

        ### down-sampling processing
        B, N, C, H, W = self.var_L.size()
        self.var_L_input = self.var_L.view(B, N * C, H, W)
        self.var_L_input = F.interpolate(self.var_L_input, size=(H // 4, W // 4), mode='bilinear')
        self.var_L_input = self.var_L_input.view(B, N, C, H // 4, W // 4)

        if need_GT:
            self.real_H = data['GT'].to(self.device)

    def set_params_lr_zero(self):
        # fix normal module
        self.optimizers[0].param_groups[0]['lr'] = 0

    def optimize_parameters(self, step):
        if self.opt['train']['ft_tsa_only'] and step < self.opt['train']['ft_tsa_only']:
            self.set_params_lr_zero()

        self.optimizer_G.zero_grad()
        self.fake_H, self.fake_H2, self.ill, self.ill2 = self.netG(self.var_L_input)
        l_pix1 = self.l_pix_w * self.cri_pix(self.fake_H, self.real_H)
        l_pix2 = self.l_pix_w * self.cri_pix(self.fake_H2, self.real_H) * 0.2
        ltv_loss_pix1 = self.cri_ltv.forward(self.ill, self.real_H.clone(), 1.0) * self.l_pix_w

        l_pix = l_pix1 + l_pix2 + ltv_loss_pix1
        self.log_dict['l_pix1'] = l_pix1.clone().item()
        self.log_dict['l_pix2'] = l_pix2.clone().item()
        self.log_dict['l_pix1_ltv'] = ltv_loss_pix1.clone().item()
        l_pix.backward()
        self.optimizer_G.step()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H, self.fake_H2, self.ill, self.ill2 = self.netG(self.var_L_input)

        self.netG.train()

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_GT=True):
        out_dict = OrderedDict()
        center = 5 // 2
        out_dict['rlt3'] = torch.div(self.var_L[:, center, :, :, :].detach()[0].float(),
                                     self.real_H.detach()[0].float() + 1e-4).cpu()
        out_dict['ill'] = self.ill.detach()[0].float().cpu()
        out_dict['LQ'] = self.var_L.detach()[0].float().cpu()
        out_dict['rlt'] = self.fake_H.detach()[0].float().cpu()
        out_dict['sr'] = self.fake_H2.detach()[0].float().cpu()
        out_dict['rlt2'] = self.fake_H.detach()[0].float().cpu()
        if need_GT:
            out_dict['GT'] = self.real_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)
