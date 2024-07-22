"""
INTR model and loss.
"""
import torch
from torch import nn
import torch.nn.functional as F

import random
from .backbone import build_backbone
from .transformer import build_transformer
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)


class INTR(nn.Module):
    """ This is the INTR module that performs explainable image classification """
    def __init__(self, transformer, args, num_queries):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py (no pos_embed in decoder)
            num_queries: number of classes in the dataset
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.args = args
        hidden_dim = transformer.d_model

        # INTR is the proposed model and INTR-FC is the baseline model.
        if args.task=='INTR':

            self.query_transform = nn.Linear(hidden_dim, 1)
        elif args.task=='INTR-FC': 
            self.query_classifiers = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_queries)])
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # self.backbone = backbone

    def forward(self, samples):

        """  The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]

            It returns the following elements:
               - "out": it is a dictnary which currently contains all logit values for for all queries.
                                Shape= [batch_size x num_queries x 1]
               - "encoder_output": it is the output of the transformer encoder which is basically feature map. 
                                Shape= [batch_size x num_features x height x weight]
               - "hs": it is the output of the transformer decoder. These are learned class specific queries. 
                                Shape= [dec_layers x batch_size x num_queries x num_features]
               - "attention_scores": it is attention weight corresponding to each pixel in the encoder  for all heads. 
                                Shape= [dec_layers x batch_size x num_heads x num_queries x height*weight]
               - "avg_attention_scores": it is attention weight corresponding to each pixel in the encoder for avg of all heads. 
                                Shape= [dec_layers x batch_size x num_queries x height*weight]

        """

        hs, encoder_output, attention_scores, avg_attention_scores = self.transformer(samples, self.query_embed.weight)

        if self.args.task=='INTR':
            query_logits = self.query_transform(hs[-1])
            query_logits=query_logits.squeeze(dim=-1)
            out = {'query_logits': query_logits}

        elif self.args.task=='INTR-FC':
            query_logits = []
            for i in range(self.num_queries):
                query_logit = self.query_classifiers[i](hs[-1][:, i, :]) # Output shape: [batch_size, 1]
                query_logits.append(query_logit.view(-1))
            query_logits = torch.stack(query_logits, dim=1)
            out = {'query_logits': query_logits}
        else:
            print ("Enter proper task, choices: ['INTR', 'INTR-FC']")
            exit()

        return out, encoder_output, hs, torch.stack(attention_scores), torch.stack(avg_attention_scores)


class SetCriterion(nn.Module):
    """ This class computes the loss for INTR.
    The process happens in two steps:
        1) In INTR, we use shared class agnostic present vector to obtain logis.
        2) In INTR-FC, we use one class agnostic present vector to for one query to obtain logis.
    """
    def __init__(self, args, weight_dict, losses, model):
        """ Create the criterion.
        Parameters:
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            losses: list of all the losses to be applied. See get_loss for list of available losses. Currently only one type of loss.
        """
        super().__init__()
        self.args = args
        self.weight_dict = weight_dict
        self.losses = losses
        self.model = model

    def loss_class(self, outputs, targets, model):
        """Classification loss (NLL)
        targets dicts must contain the key "image_label".
        """
        assert 'query_logits' in outputs
        query_logits = outputs['query_logits']
        device = query_logits.device

        target_classes = torch.cat([t['image_label'] for t in targets]) 
        
        criterion = torch.nn.CrossEntropyLoss()
        loss_classes=criterion(query_logits, target_classes)

        losses = {'loss_cls': loss_classes}

        return losses


    def get_loss(self, loss, outputs, targets, model):
        loss_map = {
            'class': self.loss_class,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, model)

    def forward(self, outputs, targets, model):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format.
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied. Here we have used only CE loss.
        """
        if self.args.task=='INTR' or self.args.task=='INTR-FC': 

            losses = {}
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, model))
        else:
            # Compute all the requested losses
            losses = {}
            for loss in self.losses:
                losses.update(self.get_loss(loss, outputs, targets, model))

        return losses

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    """
    In INTR, each query is responsible for learning class specific information.
    So, the `num_queries` here is actually the number of classes in the dataset.
    """

    if args.dataset_name== 'cub':
        args.num_queries=200

    device = torch.device(args.device)

    transformer = build_transformer(args)

    model = INTR(
        transformer,
        args,
        num_queries=args.num_queries,)

    # Here, we used only one type of loss i.e., CE.
    # In general each type of loss will be associated with some weight.
    weight_dict = {'loss_cls': args.loss_cls_coef}
    losses=['class']

    criterion = SetCriterion(args, weight_dict=weight_dict,
                            losses=losses, model=model)
    criterion.to(device)

    return model, criterion
