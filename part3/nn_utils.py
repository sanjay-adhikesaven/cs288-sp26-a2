"""
Neural network utilities for Transformer implementation.
Contains basic building blocks: softmax, cross-entropy, gradient clipping, token accuracy, perplexity.
"""
import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.
    
    Implements the numerically stable softmax:
        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    # Implement numerically stable softmax. You can re-use the same one 
    # used in part 2. But for this problem, you need to implement a numerically stable version to pass harder tests.
    
    m, _ = x.max(dim = dim, keepdim=True)
    exp_x = torch.exp(x - m)

    return exp_x/exp_x.sum(dim = dim, keepdim = True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Unnormalized log probabilities of shape (N, C) where N is batch size
                and C is number of classes
        targets: Ground truth class indices of shape (N,)
    
    Returns:
        Scalar tensor containing the mean cross-entropy loss
    """
    m, _ = logits.max(dim = -1, keepdim=True)
    log_probs = (logits - m) - torch.log(torch.exp(logits -m).sum(dim = -1, keepdim = True))

    nll = -log_probs.gather(dim = 1, index = targets.unsqueeze(1))

    return nll.mean()


def gradient_clipping(parameters, max_norm: float) -> Tensor:
    """
    Clip gradients of parameters by global norm.
    
    Implements gradient clipping by norm:
        if total_norm > max_norm:
            grad = grad * max_norm / total_norm
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns:
        The total norm of the gradients before clipping
    """
    
    total_grad = 0.0
    for p in parameters:
        if p.grad is not None:
            total_grad += torch.sum(p.grad**2)
    
    total_norm = total_grad.sqrt()

    if total_norm.item() > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(max_norm / total_norm)
    
    return total_norm




def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute token-level accuracy for language modeling.
    
    Computes the fraction of tokens where the predicted token (argmax of logits)
    matches the target token, ignoring positions where target equals ignore_index.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing accuracy (default: -100)
    
    Returns:
        Scalar tensor containing the accuracy (between 0 and 1)
    
    Example:
        >>> logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> token_accuracy(logits, targets)
        tensor(1.)  # All predictions correct: argmax gives [0, 1, 2]
        
        >>> logits = torch.tensor([[2.0, 1.0], [0.1, 3.0], [1.0, 0.5]])
        >>> targets = torch.tensor([1, 1, 0])
        >>> token_accuracy(logits, targets)
        tensor(0.6667)  # 2 out of 3 correct
    """
    predictions = logits.argmax(dim = -1)
    valid_mask = (targets != ignore_index)

    return (predictions[valid_mask] == targets[valid_mask]).float().mean()





def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute perplexity for language modeling.
    
    Perplexity is defined as exp(cross_entropy_loss). It measures how well the
    probability distribution predicted by the model matches the actual distribution
    of the tokens. Lower perplexity indicates better prediction.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing perplexity (default: -100)
    
    Returns:
        Scalar tensor containing the perplexity (always >= 1)
    
    Example:
        >>> # Perfect predictions (one-hot logits matching targets)
        >>> logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(1.0001)  # Close to 1 (perfect)
        
        >>> # Uniform predictions (high uncertainty)
        >>> logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(3.)  # Equal to vocab_size (worst case for uniform)
    """
    
    m, _ = logits.max(dim = -1, keepdim=True)
    log_probs = (logits - m) - torch.log(torch.exp(logits -m).sum(dim = -1, keepdim = True))

    targets_ignored = targets.masked_fill(targets == ignore_index, 0)
    nll = -log_probs.gather(dim = 1, index = targets_ignored.unsqueeze(1))

    return torch.exp(nll[targets != ignore_index].mean())

