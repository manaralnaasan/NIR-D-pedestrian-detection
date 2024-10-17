import torch
from model import RAFFNet
from dataset import NIRDepthDataset, transform
from utils import dice_loss, smooth_l1_loss, gaussian_heatmap, expand_to_mask, extract_bounding_boxes

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        for nir_img, depth_img, bboxes in train_loader:
            nir_img, depth_img = nir_img.to(device), depth_img.to(device)
            bboxes = bboxes.to(device)
            
            optimizer.zero_grad()
            output = model(nir_img, depth_img)
            
            pred_heatmap = gaussian_heatmap(bboxes, output.shape[-2:])
            binary_mask = expand_to_mask(pred_heatmap)
            pred_bbox = extract_bounding_boxes(binary_mask)
            
            seg_loss = dice_loss(output, binary_mask)
            bbox_loss = smooth_l1_loss(pred_bbox, bboxes)
            total_loss = seg_loss + bbox_loss
            
            total_loss.backward()
            optimizer.step()

# Define dataset and dataloader
train_dataset = NIRDepthDataset(nir_dir='path_to_nir', depth_dir='path_to_depth', xml_dir='path_to_xml', transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model and optimizer
model = RAFFNet().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

# Train the model
train_model(model, train_loader, optimizer)
