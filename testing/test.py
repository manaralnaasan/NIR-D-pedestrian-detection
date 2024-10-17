import torch

def test_model(model, test_loader):
    model.eval()
    total_seg_loss = 0
    total_bbox_loss = 0
    dice_scores = []

    with torch.no_grad():
        for nir_img, depth_img, bboxes in test_loader:
            nir_img, depth_img = nir_img.to(device), depth_img.to(device)
            bboxes = bboxes.to(device)

            # Forward pass
            output = model(nir_img, depth_img)

            # Create Gaussian heatmap for center calculation
            pred_heatmap = gaussian_heatmap(bboxes, output.shape[-2:])
            binary_mask = expand_to_mask(pred_heatmap)

            # Calculate losses
            seg_loss = dice_loss(output, binary_mask)
            bbox_loss = smooth_l1_loss(extract_bounding_boxes(binary_mask), bboxes)

            # Accumulate losses
            total_seg_loss += seg_loss.item()
            total_bbox_loss += bbox_loss.item()
            dice_scores.append(dice_score(output, binary_mask).item())

    avg_seg_loss = total_seg_loss / len(test_loader)
    avg_bbox_loss = total_bbox_loss / len(test_loader)
    avg_dice_score = np.mean(dice_scores)

    print(f"Test Segmentation Loss: {avg_seg_loss:.4f}")
    print(f"Test Bounding Box Loss: {avg_bbox_loss:.4f}")
    print(f"Test Dice Score: {avg_dice_score:.4f}")
