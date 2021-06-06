# -*- encoding: utf-8 -*-
# @File    :   loss.py
# @Time    :   2021/06/06 06:11:22
# @Author  :   Ugur Gudelek
# @Contact :   ugurgudelek@gmail.com
# @Desc    :   None

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# based on : https://kornia.readthedocs.io/en/v0.1.2/_modules/torchgeometry/losses/dice.html
def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
) -> torch.Tensor:

    if not isinstance(labels, torch.Tensor):
        raise TypeError(
            "Input labels type is not a torch.Tensor. Got {}".format(
                type(labels)))

    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}".format(
                labels.dtype))

    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))

    shape = labels.shape
    one_hot = torch.zeros((shape[0], num_classes) + shape[1:],
                          device=device,
                          dtype=dtype)

    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


def dice_loss(input: torch.Tensor,
              target: torch.Tensor,
              reduction='mean',
              eps: float = 1e-8) -> torch.Tensor:
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    According to [1], we compute the Sørensen-Dice Coefficient as follows:
    .. math::
        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}
    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.
    the loss, is finally computed as:
    .. math::
        \text{loss}(x, class) = 1 - \text{Dice}(x, class)
    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Args:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, H, W)` where C = number of classes.
        labels (torch.Tensor): labels tensor with shape :math:`(N, H, W)` where each value
          is :math:`0 ≤ targets[i] ≤ C−1`.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.
    Return:
        torch.Tensor: the computed loss.
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = dice_loss(input, target)
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if not len(input.shape) == 4:
        raise ValueError(
            "Invalid input shape, we expect BxNxHxW. Got: {}".format(
                input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError(
            "input and target shapes must be the same. Got: {} and {}".format(
                input.shape, target.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".
            format(input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target,
                                           num_classes=input.shape[1],
                                           device=input.device,
                                           dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(input_soft + target_one_hot, dims)

    dice_score = 2.0 * intersection / (cardinality + eps)

    loss = 1.0 - dice_score

    if reduction == 'sum':
        return torch.sum(loss)
    if reduction == 'mean':
        return torch.mean(loss)
    return loss


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    According to [1], we compute the Sørensen-Dice Coefficient as follows:
    .. math::
        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}
    Where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.
    the loss, is finally computed as:
    .. math::
        \text{loss}(x, class) = 1 - \text{Dice}(x, class)
    Reference:
        [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Args:
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.
    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
        >>> N = 5  # num_classes
        >>> criterion = DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self, reduction='mean', eps: float = 1e-8) -> None:
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.eps: float = eps

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return dice_loss(input, target, self.reduction, self.eps)


def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input (torch.Tensor): logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target (torch.Tensor): labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.
    Return:
        torch.Tensor: the computed loss.
    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}".format(
            input.shape))

    if input.size(0) != target.size(0):
        raise ValueError(
            'Expected input batch_size ({}) to match target batch_size ({}).'.
            format(input.size(0), target.size(0)))

    n = input.size(0)
    out_size = (n,) + input.size()[2:]
    if target.size()[1:] != input.size()[2:]:
        raise ValueError('Expected target size {}, got {}'.format(
            out_size, target.size()))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}".
            format(input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1) + eps

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target,
                                           num_classes=input.shape[1],
                                           device=input.device,
                                           dtype=input.dtype)

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * torch.log(input_soft)
    loss = torch.sum(target_one_hot * focal, dim=1)

    loss = torch.mean(loss, dim=(1, 2))

    if reduction == 'sum':
        return torch.sum(loss)
    if reduction == 'mean':
        return torch.mean(loss)
    return loss


class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2018focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    Where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha (float): Weighting factor :math:`\alpha \in [0, 1]`.
        gamma (float, optional): Focusing parameter :math:`\gamma >= 0`. Default 2.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
        eps (float, optional): Scalar to enforce numerical stabiliy. Default: 1e-8.
    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Example:
        >>> N = 5  # num_classes
        >>> kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'mean'}
        >>> criterion = FocalLoss(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = criterion(input, target)
        >>> output.backward()
    """

    def __init__(self,
                 alpha: float,
                 gamma: float = 2.0,
                 reduction: str = 'none',
                 eps: float = 1e-8) -> None:
        super(FocalLoss, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = eps

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction,
                          self.eps)


def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: float = 1e-8,
) -> torch.Tensor:
    r"""Function that computes Binary Focal loss.
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        input (torch.Tensor): input data tensor with shape :math:`(N, 1, *)`.
        target (torch.Tensor): the target tensor with shape :math:`(N, 1, *)`.
        alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`. Default: 0.25.
        gamma (float): Focusing parameter :math:`\gamma >= 0`. Default: 2.0.
        reduction (str, optional): Specifies the reduction to apply to the. Default: 'none'.
        eps (float): for numerically stability when dividing. Default: 1e-8.
    Returns:
        torch.tensor: the computed loss.
    Examples:
        >>> num_classes = 1
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[[6.325]]],[[[5.26]]],[[[87.49]]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(4.6052)
    """

    if not isinstance(input, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))

    if not len(input.shape) >= 2:
        raise ValueError("Invalid input shape, we expect BxCx*. Got: {}".format(
            input.shape))

    if input.size(0) != target.size(0):
        raise ValueError(
            'Expected input batch_size ({}) to match target batch_size ({}).'.
            format(input.size(0), target.size(0)))

    probs = torch.sigmoid(input)
    target = target.unsqueeze(dim=1)
    loss_tmp = -alpha * torch.pow(
        (1.0 - probs + eps),
        gamma) * target * torch.log(probs + eps) - (1 - alpha) * torch.pow(
            probs + eps, gamma) * (1.0 - target) * torch.log(1.0 - probs + eps)

    loss_tmp = loss_tmp.squeeze(dim=1)

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(
            "Invalid reduction mode: {}".format(reduction))
    return loss


class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.
    According to :cite:`lin2017focal`, the Focal loss is computed as follows:
    .. math::
        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)
    where:
       - :math:`p_t` is the model's estimated probability for each class.
    Args:
        alpha (float): Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma (float): Focusing parameter :math:`\gamma >= 0`.
        reduction (str, optional): Specifies the reduction to apply to the
         output: ‘none’ | ‘mean’ | ‘sum’. ‘none’: no reduction will be applied,
         ‘mean’: the sum of the output will be divided by the number of elements
         in the output, ‘sum’: the output will be summed. Default: ‘none’.
    Shape:
        - Input: :math:`(N, 1, *)`.
        - Target: :math:`(N, 1, *)`.
    Examples:
        >>> N = 1  # num_classes
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = BinaryFocalLossWithLogits(**kwargs)
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self,
                 alpha: float,
                 gamma: float = 2.0,
                 reduction: str = 'none') -> None:
        super(BinaryFocalLossWithLogits, self).__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: float = 1e-8

    def forward(self, input: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(input, target, self.alpha,
                                             self.gamma, self.reduction,
                                             self.eps)


class EnhancedMixingLoss(torch.nn.Module):

    def __init__(self, gamma=1.1, alpha=0.48, reduction='mean'):
        super(EnhancedMixingLoss, self).__init__()
        self.focal_loss = FocalLoss(gamma=gamma,
                                    alpha=alpha,
                                    reduction=reduction)
        self.dice_loss = DiceLoss(reduction=reduction)
        self.reduction = reduction

    def forward(self, logit, target):
        fcloss = self.focal_loss(logit, target)
        dcloss = self.dice_loss(logit, target)

        loss = fcloss - torch.log(dcloss)

        return loss