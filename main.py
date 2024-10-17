import argparse
from train import train_model
from test import test_model
from dataset import NIRDepthDataset
from model import RAFFNet
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(description="NIR-D Pedestrian Detection and Segmentation")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RAFFNet().to(device)

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    if args.train:
        train_loader = NIRDepthDataset('path/to/train/nir_images', 'path/to/train/depth_images', 'path/to/train/xml_annotations')
        train_model(model, train_loader, optimizer)

    if args.test:
        test_loader = NIRDepthDataset('path/to/test/nir_images', 'path/to/test/depth_images', 'path/to/test/xml_annotations')
        test_model(model, test_loader)
