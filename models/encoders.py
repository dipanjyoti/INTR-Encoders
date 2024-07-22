import torch
import timm
from transformers import ViTFeatureExtractor
from transformers import DeiTConfig, DeiTModel
from transformers import AutoImageProcessor, Dinov2Model


def get_encoder(args):

    encoder = timm.create_model(args.encoder, pretrained=False, num_classes=0,  )

    return encoder
