import torch
from model import RAFFNet
from dataset import NIRDepthDataset, transform
from utils import dice_loss, smooth_l1_loss, gaussian_heatmap, expand_to_mask, extract_bounding_boxes

def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for nir_img, depth_img, bboxes in test_loader:
            nir_img, depth_img = nir_img.to(device), depth_img.to(device)
            bboxes = bboxes.to(device)
            
            output = model(nir_img, depth_img)
            pred_heatmap = gaussian_heatmap(bboxes, output.shape[-2:])
            binary_mask = expand_to_mask(pred_heatmap)
            pred_bbox = extract_bounding_boxes(binary_mask)
            
            seg_loss = dice_loss(output, binary_mask)
            bbox_loss = smooth_l1_loss(pred_bbox, bboxes)
            print(f'Seg Loss: {seg_loss.item()}, BBox Loss: {bbox_loss.item()}')

test_dataset = NIRDepthDataset(nir_dir='path_to_test_nir', depth_dir='path_to_test_depth', xml_dir='path_to_test_xml', transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load trained model
model = RAFFNet().to(device)
model.load_state_dict(torch.load('model.pth'))

# Test the model
test_model(model, test_loader)
