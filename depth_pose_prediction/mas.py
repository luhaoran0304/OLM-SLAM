import torch
import torch.autograd as ag
import numpy as np
import torch.nn.functional as F
from torch import Tensor, nn
from typing import Any, Dict, List
import cv2


class MAS:
    def __init__(self, scales):
        self.depth_cur_imp, self.depth_old_imp = None, None
        self.depth_cur_param, self.depth_old_param = None, None
        self.pose_cur_imp, self.pose_old_imp = None, None
        self.pose_cur_param, self.pose_old_param = None, None
        self.depth_model = None
        self.pose_model = None
        self.initialized = False
        self.n_observed = 0
        self.scales = scales
        self.frame_ids = (0, -1, 1)
        self.device = 'cuda'
        self.bs = 0
        self.avg_method = 'avg'
        self.encoder = None

    def __call__(self, outputs: Dict[Any, Tensor], target: Tensor,
                 depth_model: nn.Module, pose_model: nn.Module, encoder: nn.Module):
        if not self.initialized:
            self.init_loss(depth_model, pose_model, encoder)
            self.initialized = True
        loss = self.calculate_total_loss(outputs, target)
        self.bs = len(loss)
        depth_ll_loss = self.calc_depth_loss(loss)
        pose_ll_loss = self.calc_pose_loss(loss)
        ll_loss = depth_ll_loss + pose_ll_loss
        self.restore_imp(self.depth_cur_imp, self.pose_cur_imp)
        self.n_observed += self.bs
        return ll_loss

    def init_loss(self, depth_model: nn.Module, pose_model: nn.Module, encoder: nn.Module) -> None:
        assert depth_model is not None
        assert pose_model is not None
        self.depth_model = depth_model
        self.pose_model = pose_model
        self.encoder = encoder
        # save params from last task
        self.depth_cur_param = list(depth_model.parameters())
        self.depth_old_param = [t.data.clone() for t in self.depth_cur_param]
        # 0 importance if not set, ensure same device if loaded
        self.depth_cur_imp = [torch.zeros_like(t.data) for t in self.depth_cur_param]
        if self.depth_old_imp is None:
            self.depth_old_imp = [torch.zeros_like(t.data) for t in self.depth_cur_param]
        elif self.depth_old_imp[0].device != self.depth_old_param[0].device:
            self.depth_old_imp = [ow.to(self.depth_old_param[0].device) for ow in self.depth_old_imp]

        # save params from last task
        self.pose_cur_param = list(pose_model.parameters())
        self.pose_old_param = [t.data.clone() for t in self.pose_cur_param]
        # 0 importance if not set, ensure same device if loaded
        self.pose_cur_imp = [torch.zeros_like(t.data) for t in self.pose_cur_param]
        if self.pose_old_imp is None:
            self.pose_old_imp = [torch.zeros_like(t.data) for t in self.pose_cur_param]
        elif self.pose_old_imp[0].device != self.pose_old_param[0].device:
            self.pose_old_imp = [ow.to(self.pose_old_param[0].device) for ow in self.pose_old_imp]

    def calc_depth_loss(self, loss):
        '''Collect weights for current task and penalize with those from previous tasks.'''
        gs = self.calculate_grad(loss, self.depth_model)
        losses = 0.0
        for imp, param, old, ow, w in zip(gs, self.depth_cur_param, self.depth_old_param, self.depth_old_imp,
                                          self.depth_cur_imp):
            if self.avg_method == 'avg':
                w.data = (w * self.n_observed + imp) / (self.n_observed + self.bs)
            elif self.avg_method == 'none':
                w.data = imp
            losses += (ow * (param - old) ** 2).sum()
        return losses

    def calc_pose_loss(self, loss):
        '''Collect weights for current task and penalize with those from previous tasks.'''
        gs = self.calculate_grad(loss, self.pose_model)
        losses = 0.0
        for imp, param, old, ow, w in zip(gs, self.pose_cur_param, self.pose_old_param, self.pose_old_imp,
                                          self.pose_cur_imp):
            if self.avg_method == 'avg':
                w.data = (w * self.n_observed + imp) / (self.n_observed + self.bs)
            elif self.avg_method == 'none':
                w.data = imp
            losses += (ow * (param - old) ** 2).sum()
        return losses

    def restore_imp(self, depth_state: List, pose_state: List):
        self.depth_old_imp = depth_state
        self.pose_old_imp = pose_state

    def calculate_grad(self, loss, model):
        gradients = ag.grad(loss, model.parameters(), create_graph=True)
        return [g.abs() for g in gradients]

    def calculate_total_loss(self, outputs, target):
        total_loss = torch.zeros(1, device=self.device)
        for scale in self.scales:
            kl_losses = []
            phash_losses = []
            for frame_id in self.frame_ids[1:]:
                pred = outputs['rgb', frame_id, scale]
                kl_losses.append(self.calculate_kl(pred, target))
                phash_losses.append(self.calculate_phash(pred, target))

            kl_loss = sum(kl_losses) / len(kl_losses)
            phash_loss = sum(phash_losses) / len(phash_losses)

            kl_weight = 1 / 2 ** (3 - scale)
            phash_weight = 1 / 2 ** (3 - scale)
            ll_loss = kl_weight * kl_loss + \
                      phash_loss * phash_weight


            total_loss += ll_loss
        total_loss /= len(self.scales)
        return total_loss

    def calculate_kl(self,
                     pred: Tensor,
                     target: Tensor,
                     ):
        b = pred.shape[0]
        kl_loss = 0.0
        for i in range(0, b):

            pred_r = F.softmax(pred[i, 0, :, :].view(-1), dim=0)
            target_r = F.softmax(target[i, 0, :, :].view(-1), dim=0)
            pred_g = F.softmax(pred[i, 1, :, :].view(-1), dim=0)
            target_g = F.softmax(target[i, 1, :, :].view(-1), dim=0)
            pred_b = F.softmax(pred[i, 2, :, :].view(-1), dim=0)
            target_b = F.softmax(target[i, 2, :, :].view(-1), dim=0)

            kl_r = torch.sum(pred_r * (torch.log(pred_r) - torch.log(target_r)))
            kl_g = torch.sum(pred_g * (torch.log(pred_g) - torch.log(target_g)))
            kl_b = torch.sum(pred_b * (torch.log(pred_b) - torch.log(target_b)))
            kl_loss += (kl_r + kl_g + kl_b) / 3
        return kl_loss

    def calculate_phash(self, pred, target):
        p1 = self.pHash(pred)
        p2 = self.pHash(target)
        ham = self.Hamming_distance(p1, p2)
        return ham

    def pHash(self, image):
        image_np = image.detach().cpu().numpy()
        if image_np.ndim == 4:
            image_np = image_np[0]
        image_np = image_np.transpose(1, 2, 0)
        if image_np.ndim == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        image_np = cv2.resize(image_np, (32, 32), interpolation=cv2.INTER_CUBIC)

        dct = cv2.dct(np.float32(image_np))
        dct_roi = dct[0:8, 0:8]
        average = np.mean(dct_roi)
        hash = []
        for i in range(dct_roi.shape[0]):
            for j in range(dct_roi.shape[1]):
                if dct_roi[i, j] > average:
                    hash.append(1)
                else:
                    hash.append(0)
        hash_np = np.array(hash, dtype=np.float32).reshape(1, -1)
        return hash_np

    def Hamming_distance(self, hash1: Tensor, hash2: Tensor) -> int:
        assert hash1.shape == hash2.shape
        hamming_dist = (hash1 != hash2).sum().item()
        return hamming_dist