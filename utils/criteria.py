import torch.nn as nn
import torch

loss_names = ['l1', 'l2']


class MaskedMSELoss(nn.Module):

    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "MSE_Loss: inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss


class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, pred, target):
        valid_mask = (target > 0).detach()
        diff = torch.abs(target - pred)
        diff = diff[valid_mask]
        cond = diff < self.beta
        self.loss = 2 * torch.where(cond, 0.5 * diff**2, diff - 0.5 * self.beta).mean()
        return self.loss


class Distance(nn.Module):

    def __init__(self):
        super(Distance, self).__init__()

    def forward(self, student, teacher, gt):
        assert student.dim() == teacher.dim() and student.dim() == gt.dim(), "KL_DIV: inconsistent dimensions"
        valid_mask = (gt > 0).detach()
        valid_mask = valid_mask.expand(student.shape)
        diff = student - teacher
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss


class FeatureDistance(nn.Module):

    def __init__(self):
        super(FeatureDistance, self).__init__()

    def forward(self, student, teacher):
        assert student.dim() == teacher.dim(), "KL_DIV: inconsistent dimensions"
        diff = student - teacher
        self.loss = (diff**2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):

    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target, weight=None):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
