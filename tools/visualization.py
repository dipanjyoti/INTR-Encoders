# Copyright (c) 

import os
import cv2 
import json
import time
import math
import shutil
import random
import argparse
import datetime

import numpy as np
from PIL import Image
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from models import build_model
from datasets import build_dataset
from engine import train_one_epoch
from scipy.ndimage import gaussian_filter
# from engine import evaluate, train_one_epoch

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1.00e-4, type=float) 
    parser.add_argument('--lr_backbone', default=1.00e-5, type=float)
    parser.add_argument('--min_lr', default=1.00e-7, type=float)
    parser.add_argument('--batch_size', default=1, type=int, choices=[1])
    parser.add_argument('--weight_decay', default=1e-6, type=float)
    parser.add_argument('--epochs', default=100, type=int) 
    parser.add_argument('--lr_drop', default=80, type=int)
    parser.add_argument('--lr_scheduler', default="StepLR", type=str, choices=["StepLR", "CosineAnnealingLR"])
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Data Transform parameters for swin and vit
    parser.add_argument('--AUG_COLOR_JITTER', default=0.4, type=float,
                        help="Help")
    parser.add_argument('--AUG_AUTO_AUGMENT', default="rand-m9-mstd0.5-inc1", type=str,
                        help="Help")
    parser.add_argument('--AUG_REPROB', default=0.25, type=float,
                        help="Help")
    parser.add_argument('--AUG_REMODE', default="pixel", type=str,
                        help="Help")
    parser.add_argument('--AUG_RECOUNT', default=1, type=int,
                        help="Help")
    parser.add_argument('--DATA_INTERPOLATION', default="bicubic", type=str,
                        help="Help") 
    parser.add_argument('--TEST_CROP', default=True, type=bool,
                        help="")

    # * Encoder
    parser.add_argument('--data_transform', default="vit", type=str, choices=["detr", "swin", "vit"])
    parser.add_argument('--img_size', default=518, type=int, choices=[192, 224, 512],
                        help="Number of encoder output dimension")
    parser.add_argument('--encoder_op_dim', default=384, type=int, choices=[384, 768, 786, 1024],
                        help="Number of encoder output dimension")
    parser.add_argument('--encoder', default='vit_small_patch16_224.dino', type=str, choices=['vit_base_patch16_224.augreg_in21k', 
                                            'vit_huge_patch14_224.orig_in21k', 'vit_small_patch16_224.dino',])

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--task', default="INTR", type=str, choices=["INTR", "INTR-FC"])
    parser.add_argument('--test', default="val", type=str, choices=["val", "test"])

    # visualization parameters
    parser.add_argument('--class_index', default=15, type=int,
                        help="a class number to visuaization")
    parser.add_argument('--dec_layer_index', default=5, type=int,
                        help="a layer number to visuaization")
    parser.add_argument('--top_q', default=5, type=int,
                        help="top most similar queries")
    parser.add_argument('--check_heads', default=1, type=int,
                        help="print all heads for visualization")
    parser.add_argument('--check_queries', default=0, type=int,
                        help="print all heads of similar queries for visualization")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer") #default=0.1
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=200, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # * Loss
    parser.add_argument('--loss_cls_coef', default=1, type=float)

    # dataset parameters
    parser.add_argument('--dataset_name', default='cub')
    parser.add_argument('--dataset_path', default='', type=str)
    parser.add_argument('--output_dir', default='output',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def get_image_filename(args, filename):
    image_path=os.path.join(args.dataset_path, args.dataset_name + '/' + args.test)
    image_filename = os.path.join(image_path, filename)
    return image_filename

def combine_images(path, pred_class):
    images = [os.path.join(path, image) for image in os.listdir(path) if image.endswith('.png')]
    imgs = [Image.open(image) for image in images]
    widths, heights = zip(*(img.size for img in imgs))

    total_width = sum(widths)
    max_height = max(heights)
    merged_image = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for img in imgs:
        merged_image.paste(img, (x_offset, 0))
        x_offset += img.width
    merged_image.save(path + "/" + "concatenated_"+str(bool(pred_class))+".png")

def HeadLevelSimilarity(attention_score):
    min_indexes=[]
    for h in range(attention_score.shape[1]):
        head=attention_score[:, h,: , :]
        query_first=head[:, similar_queries[0], :]
        l1_distances=[]
        for r in range(1, len(similar_queries)):
            query_rest=head[:, similar_queries[r], :]
            l1_distance = torch.sum(torch.abs(query_first - query_rest))
            l1_distances.append(float(l1_distance.item()))
        min_index = l1_distances.index(min(l1_distances))
        min_indexes.append(min_index)
    return min_indexes

def eigen_smooth_heatmap(heatmap, sigma):
    median = cv2.medianBlur(heatmap,150)
    return median
    reshaped_heatmap = heatmap.reshape(-1, heatmap.shape[-1])

    cov_matrix = np.cov(reshaped_heatmap.T)

    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[sorted_indices]

    filter_ = np.exp(-(eigenvalues / np.max(eigenvalues))**2 / (2 * sigma**2))
    smoothed_heatmap = gaussian_filter(reshaped_heatmap.T, sigma=sigma, mode='reflect')

    filtered_heatmap = smoothed_heatmap.T * filter_
    filtered_heatmap = filtered_heatmap.reshape(heatmap.shape)

    return filtered_heatmap

def SuperImposeHeatmap(attention, input_image):
    alpha=0.5
    avg_heatmap_resized = cv2.resize(attention, (input_image.shape[1], input_image.shape[0]), interpolation=cv2.INTER_CUBIC)
    #avg_heatmap_resized = eigen_smooth_heatmap(avg_heatmap_resized, sigma)
    avg_normalized_heatmap = (avg_heatmap_resized - np.min(avg_heatmap_resized)) / (np.max(avg_heatmap_resized) - np.min(avg_heatmap_resized))
    heatmap = (avg_normalized_heatmap * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.medianBlur(heatmap,15)
    heatmap =  cv2.GaussianBlur(heatmap, (15, 15), 0)
    result = (input_image *alpha  + heatmap * (1-alpha)).astype(np.uint8)
    # result = cv2.resize(result, (0,0), fx=0.3, fy=0.3)
    return result

def visualize_heads(attention_score_query, avg_attention_score_query, encoder_output, des_dir, pred_class, image_file, img_id): #, input_size

    if not os.path.exists(os.path.join(des_dir, f"heads")):
        os.mkdir(os.path.join(des_dir, f"heads"))
    input_image = cv2.imread(image_file)

    input_image = cv2.resize(input_image, (0,0), fx=0.3, fy=0.3) 
    des_image_file=os.path.join(des_dir+ '/' + f"heads" + "/"  + str(img_id) + '.png')
    cv2.imwrite(des_image_file, input_image)

    for h in range(attention_score_query.shape[1]):

        heatmap_head=attention_score_query[:, h, :].reshape(encoder_output.shape[2], encoder_output.shape[3]).detach().cpu().numpy()
        result=SuperImposeHeatmap(heatmap_head, input_image)

        filename = des_dir + "/" + f"heads" + "/" + f"result_head_{h}.png"
        cv2.imwrite(filename, result)

        if h==7: # This is for avg attention score
            avg_heatmap_head=avg_attention_score_query.reshape(encoder_output.shape[2], encoder_output.shape[3]).detach().cpu().numpy()
            result=SuperImposeHeatmap(avg_heatmap_head, input_image)
            filename = des_dir + "/" + f"heads" + "/" + f"avg_head.png" 
            cv2.imwrite(filename,result)

    combine_images(des_dir + "/" + f"heads", pred_class)


def visualize_queries(attention_score, avg_attention_score, encoder_output, des_dir, pred_class, image_file, img_id, similar_queries): 

    if not os.path.exists(os.path.join(des_dir, f"head_avg")):
        os.mkdir(os.path.join(des_dir, f"head_avg"))
    input_image = cv2.imread(image_file)
    input_image = cv2.resize(input_image, (0,0), fx=0.3, fy=0.3)

    for q in (similar_queries):
        avg_attention_score_query=avg_attention_score[ :, q, :]
        avg_heatmap=avg_attention_score_query.reshape(encoder_output.shape[2], encoder_output.shape[3]).detach().cpu().numpy()
        result=SuperImposeHeatmap(avg_heatmap, input_image)

        filename = des_dir + "/" + f"head_avg" + "/" + f"avg_head_{q}.png" 
        cv2.imwrite(filename, result)

    # min_indexes=HeadLevelSimilarity(attention_score)
    for h in range (attention_score.shape[1]):
        if not os.path.exists(os.path.join(des_dir, f"head_{h}")):
            os.mkdir(os.path.join(des_dir, f"head_{h}"))

        des_image_file=os.path.join(des_dir+ '/'  + f"head_{h}" + "/"  + str(img_id) + '.png')
        cv2.imwrite(des_image_file, input_image) 

        attention_score_head=attention_score[:, h,: , :]

        for q in (similar_queries):
            heatmap_query=attention_score_head[:, q, :].reshape(encoder_output.shape[2], encoder_output.shape[3]).detach().cpu().numpy()
            result=SuperImposeHeatmap(heatmap_query, input_image)

            filename = des_dir + "/" + f"head_{h}" + "/" + f"result_query_{q}.png"
            cv2.imwrite(filename, result)

        combine_images(des_dir + "/" + f"head_{h}",pred_class)


def visualization(args, filename, attention_scores, avg_attention_scores, encoder_output, targets, similar_queries, prediction):
    """
    This visualization function is for visualizing attention score for all the heads, similar queries, avg attention score etc.
        -- if check_heads is True, it will visualize attention score corresponding to different heads of an image.
        -- if check_queries is True, it will visualize attention score corresponding to all the similar queries of an image belonging to a class.
    """

    output_dir = args.output_dir
    vis_dir = "visualization"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    vis_path = os.path.join(output_dir, vis_dir)
    if not os.path.exists(vis_path):
        os.mkdir(vis_path)

    image_labels = [item['image_label'].item() for item in targets]
    
    if args.check_heads==1:
        print ("visualizing all head attention weights ...")

        attention_score=attention_scores[args.dec_layer_index, :, :, :, :]
        avg_attention_score=avg_attention_scores[args.dec_layer_index, :, :, :]

        for img in range(attention_score.shape[0]): 
            q=(image_labels[img]) 
            pred_class=int(prediction[img])

            img_id = filename.split("/")[-1][:-4]

            attention_score_query=attention_score[:, :, q, :]
            avg_attention_score_query=avg_attention_score[ :, q, :]

            if not os.path.exists(os.path.join(vis_path, str(img_id))):
                os.mkdir(os.path.join(vis_path, str(img_id)))
            des_dir=os.path.join(output_dir, vis_dir, str(img_id))

            image_file=get_image_filename(args, filename)
            visualize_heads(attention_score_query, avg_attention_score_query, encoder_output, des_dir, pred_class,  image_file, img_id)

    if args.check_queries==1: 

        print ("visualizing all query attention weights ...")

        attention_score=attention_scores[args.dec_layer_index, :, :, :, :]
        avg_attention_score=avg_attention_scores[args.dec_layer_index, :, :, :]
        for img in range(attention_score.shape[0]):
            pred_class=int(prediction[img])
            img_id = filename.split("/")[-1][:-4]
            
            if not os.path.exists(os.path.join(vis_path, str(img_id))):
                os.mkdir(os.path.join(vis_path, str(img_id)))
            des_dir=os.path.join(output_dir, vis_dir, str(img_id))

            image_file=get_image_filename(args, filename)
            visualize_queries(attention_score, avg_attention_score, encoder_output, des_dir, pred_class, image_file, img_id, similar_queries)
            
            
@torch.no_grad()
def evaluate(args, model, data_loader, device):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for samples, targets in metric_logger.log_every(data_loader, 100, header):
        samples = samples.to(device)

        filenames=targets["file_name"]
        parts = filenames[0][0].split("/")
        filename = "/".join(parts[-2:])

        image_labels=targets["image_label"] 
        image_labels = image_labels.to(device)
        targets = [{'image_label': label} for label in image_labels]

        # Here we visualize only for the given class_index
        if image_labels[0][0]==args.class_index:

            outputs, encoder_output, decoder_output, attention_scores, avg_attention_scores = model(samples)
            logits=outputs['query_logits'].flatten()


            # We find the top few similar queries produced by the model
            _, similar_queries = torch.topk(logits, k=args.top_q)
            similar_queries=similar_queries.tolist()

            # In case of incorrect prediction, and the true is not in similar_queries, we manually add it.
            if  args.class_index not in similar_queries:
                similar_queries[-1]=args.class_index


            _ , _, prediction = utils.class_accuracy(outputs, targets, topk=(1, 1))
            visualization(args, filename, attention_scores, avg_attention_scores, encoder_output, targets, similar_queries, prediction) 


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, _ = build_model(args)
    model.to(device)
    model_without_ddp = model

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu]) #, find_unused_parameters=True
        model_without_ddp = model.module

    dataset_val = build_dataset(image_set=args.test, args=args)

    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                drop_last=False, num_workers=args.num_workers)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)

    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')

        model_without_ddp.load_state_dict(checkpoint['model'])
        model_without_ddp.to(device)

    if args.eval:
        evaluate(args, model, data_loader_val, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('INTR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
