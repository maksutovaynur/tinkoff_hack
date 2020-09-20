from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from catalyst.utils.metrics.functional import preprocess_multi_label_metrics
from catalyst.utils.torch import get_activation_fn
from dataset import CAT_TYPE_NCLASSES, MERCH_TYPE_NCLASSES, TRANS_TYPE_NCLASSES


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.merchant_type_embedding = nn.Embedding(
            MERCH_TYPE_NCLASSES, params["merchant_type_emb_dim"]
        )
        self.trans_type_embedding = nn.Embedding(
            TRANS_TYPE_NCLASSES, params["trans_type_embedding"]
        )

        self.cat_type_embedding = nn.Embedding(
            CAT_TYPE_NCLASSES, params["cat_type_embedding"]
        )

        embedding_size = (
            params["merchant_type_emb_dim"]
            + params["trans_type_embedding"]
            + params["cat_type_embedding"]
            + 1
        )

        transformer_blocks = []
        for i in range(params["num_layers"]):
            transformer_block = nn.TransformerEncoderLayer(
                d_model=embedding_size,
                nhead=params["transformer_nhead"],
                dim_feedforward=params["transformer_dim_feedforward"],
                dropout=params["transformer_dropout"],
            )
            transformer_blocks.append(
                (f"transformer_block_{i}", transformer_block)
            )

        self.transformer_encoder = nn.Sequential(
            OrderedDict(transformer_blocks)
        )

        self.linear = nn.Linear(
            in_features=embedding_size, out_features=params["dense_unit"]
        )
        self.scorer = nn.Linear(
            in_features=params["dense_unit"],
            out_features=MERCH_TYPE_NCLASSES - 1,
        )

    def forward(self, features):

        merchant_type_emb = self.merchant_type_embedding(features["merchant_type"])
        trans_type_emb = self.trans_type_embedding(features["trans_type"])
        cat_type_emb = self.cat_type_embedding(features["transaction_category"])

        merchant_type_emb = merchant_type_emb * features["merchant_type_mask"].unsqueeze(-1)
        trans_type_emb = trans_type_emb * features["trans_type_mask"].unsqueeze(-1)
        cat_type_emb = cat_type_emb * features["transaction_category_mask"].unsqueeze(-1)


        embeddings = torch.cat(
            (merchant_type_emb, trans_type_emb, cat_type_emb,
             features["sum"].unsqueeze(-1),
             features["financial_account_type_cd"].unsqueeze(-1),
             features["day_of_week"].unsqueeze(-1),
             features["gender_cd"].unsqueeze(-1),
             features["age"].unsqueeze(-1), features["marital_status_desc"].unsqueeze(-1),
             features["children_cnt"].unsqueeze(-1), features["most_popular_like_category"].unsqueeze(-1),
             features["most_popular_favorite_category"].unsqueeze(-1), features["most_popular_dislike_category"].unsqueeze(-1),
             features["products_sum"].unsqueeze(-1), features["most_popular_product_chosen"].unsqueeze(-1)),
             dim=-1,
        )

        transformer_output = self.transformer_encoder(embeddings)
        pooling = torch.mean(transformer_output, dim=1)
        linear = torch.tanh(self.linear(pooling))
        merch_logits = self.scorer(linear)

        return merch_logits

def multi_label_metrics(
        outputs: torch.Tensor,
        targets: torch.Tensor,
        threshold: Union[float, torch.Tensor],
        activation: Optional[str] = None,
        eps: float = 1e-7,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes multi-label precision for the specified activation and threshold.

    Args:
        outputs (torch.Tensor): NxK tensor that for each of the N examples
            indicates the probability of the example belonging to each of
            the K classes, according to the model.
        targets (torch.Tensor): binary NxK tensort that encodes which of the K
            classes are associated with the N-th input
            (eg: a row [0, 1, 0, 1] indicates that the example is
            associated with classes 2 and 4)
        threshold (float): threshold for for model output
        activation (str): activation to use for model output
        eps (float): epsilon to avoid zero division

    Extended version of
        https://github.com/catalyst-team/catalyst/blob/master/catalyst/utils/metrics/accuracy.py#L58

    Returns:
        computed multi-label metrics
    """
    outputs, targets, _ = preprocess_multi_label_metrics(
        outputs=outputs, targets=targets
    )
    activation_fn = get_activation_fn(activation)
    outputs = activation_fn(outputs)

    outputs = (outputs > threshold).long()
    print(f'outputs.size() = {outputs.size()}')
    print(f'outputs = {outputs}')
    print(f'targets.size() = {targets.size()}')
    print(f'targets = {targets}')
    accuracy = (targets.long() == outputs.long()).sum().float() / np.prod(
        targets.shape
    )

    intersection = (outputs.long() * targets.long()).sum(axis=1).float()
    num_predicted = outputs.long().sum(axis=1).float()
    num_relevant = targets.long().sum(axis=1).float()
    union = num_predicted + num_relevant

    # Precision = ({predicted items} && {relevant items}) / {predicted items}
    precision = intersection / (num_predicted + eps * (num_predicted == 0))
    # Recall = ({predicted items} && {relevant items}) / {relevant items}
    recall = intersection / (num_relevant + eps * (num_relevant == 0))
    # IoU = ({predicted items} && {relevant items}) / ({predicted items} || {relevant items})
    iou = (intersection + eps * (union == 0)) / (union - intersection + eps)

    return accuracy, precision.mean(), recall.mean(), iou.mean()


def precision_at_k(
        actual: torch.Tensor,
        predicted: torch.Tensor,
        k: int,
):
    """
    Computes precision at cutoff k for one sample

    Args:
       actual: (torch.Tensor): tensor of length K with predicted item_ids sorted by relevance
       predicted (torch.Tensor): binary tensor that encodes which of the K
           classes are associated with the N-th input
           (eg: a row [0, 1, 0, 1] indicates that the example is
           associated with classes 2 and 4)
       k (int): parameter k of precison@k

    Returns:
       Computed value of precision@k for given sample
    """
    p_at_k = 0.0
    for item in predicted[:k]:
        if actual[item]:
            p_at_k += 1
    p_at_k /= k

    return p_at_k


def average_precision_at_k(
        actual: torch.Tensor,
        predicted: torch.Tensor,
        k: int,
) -> float:
    """
    Computes average precision at cutoff k for one sample

    Args:
      actual: (torch.Tensor): tensor of length K with predicted item_ids sorted by relevance
      predicted (torch.Tensor): binary tensor that encodes which of the K
          classes are associated with the N-th input
          (eg: a row [0, 1, 0, 1] indicates that the example is
          associated with classes 2 and 4)
      k (int): parameter k of AP@k

    Returns:
        Computed value of AP@k for given sample
    """
    ap_at_k = 0.0
    for idx, item in enumerate(predicted[:k]):
        if actual[item]:
            ap_at_k += precision_at_k(actual, predicted, k=idx + 1)
    ap_at_k /= min(k, actual.sum().cpu().numpy())

    return ap_at_k


def mean_average_precision_at_k(
        output: torch.Tensor, target: torch.Tensor, top_k: Tuple[int, ...] = (1,)
) -> List[float]:
    """
    Computes mean_average_precision_at_k at set of cutoff parameters K

    Args:
       outputs (torch.Tensor): NxK tensor that for each of the N examples
           indicates the probability of the example belonging to each of
           the K classes, according to the model.
       targets (torch.Tensor): binary NxK tensort that encodes which of the K
           classes are associated with the N-th input
           (eg: a row [0, 1, 0, 1] indicates that the example is
           associated with classes 2 and 4)
       top_k (tuple): list of parameters k at which map@k will be computed


    Returns:
       List of computed values of map@k at each cutoff k from topk
    """
    max_k = max(top_k)
    batch_size = target.size(0)

    _, top_indices = output.topk(k=max_k, dim=1, largest=True, sorted=True)

    result = []
    for k in top_k:  # loop over k
        map_at_k = 0.0
        for actual_target, predicted_items in zip(
                target, top_indices
        ):  # loop over samples
            map_at_k += average_precision_at_k(
                actual_target, predicted_items, k
            )
        map_at_k = map_at_k / batch_size
        result.append(map_at_k)

    return result
