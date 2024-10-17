import torch
from model.raffnet import RAFFNet
from data.dataset import NIRDepthDataset
from training.train import train_model
from testing.test import test_model

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load datasets
    train_dataset = NIRDepthDataset('path/to/train/nir_images', 
                                     'path/to/train/depth_images', 
                                     'path/to/train/xml_annotations')
    test_dataset = NIRDepthDataset('path/to/test/nir_images', 
                                    'path/to/test/depth_images', 
                                    'path/to/test/xml_annotations')
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model and optimizer
    model = RAFFNet().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    
    # Train the model
    train_model(model, train_loader, optimizer, num_epochs=25)
    
    # Test the model
    test_model(model, test_loader)

if __name__ == "__main__":
    main()
