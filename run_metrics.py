import matplotlib.pyplot as plt
import numpy as np
from torchtrainer.util.post_processing import logits_to_preds
import torch
from topology_metrics import ClDice
from confusion_metrics import ConfusionMatrixMetrics
from distance_metrics import DistanceMetrics
from hce_metric import HCEMetric
from pathlib import Path
from argparse import ArgumentParser
import networkx as nx
from graph_metrics import calculate_graph_distances

def run_all_metrics(scores, targets):
    if isinstance(scores, nx.classes.multigraph.MultiGraph):    
        distances = calculate_graph_distances(scores, targets)
    else:
        distances = run_image_metrics(scores, targets)

    return distances


def run_image_metrics(scores, targets):
    clDiceClass = ClDice()
    confusionMatrixClass = ConfusionMatrixMetrics()
    DistanceMetricsClass = DistanceMetrics()
    HCEMetricClass = HCEMetric()

    scores_reshaped = reshape_4_dim(scores)
    targets_reshaped = reshape_4_dim(targets)
    targets_reshaped_3_dim = reshape_3_dim(targets)
    
    metrics_dict = {}

    metrics_dict["clDice"] = clDiceClass(scores_reshaped, targets_reshaped)
    metrics_dict["confusionMatrixMetrics"] = confusionMatrixClass(scores_reshaped, targets_reshaped)
    metrics_dict["distanceMetrics"] = DistanceMetricsClass(scores_reshaped, targets_reshaped_3_dim)

    return metrics_dict

def reshape_4_dim(input):
    return input.unsqueeze(0).unsqueeze(0)

def reshape_3_dim(input):
    return input.unsqueeze(0)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--save", type=Path)
    
    args = parser.parse_args()
    scores = args.scores
    targets = args.targets
    save = args.save
    run_metrics(scores, targets)
