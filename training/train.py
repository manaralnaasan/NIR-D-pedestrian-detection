import torch

def train_model(model, train_loader, optimizer, num_epochs=25):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for nir_img, depth_img, bboxes in train_loader:
            nir_img, depth_img = nir_img.to(device), depth_img.to(device)
            bboxes = bboxes.to(device)

            optimizer.zero_grad()
            output = model(nir_img, depth_img)

            # Create Gaussian heatmap for center calculation
            pred_heatmap = gaussian_heatmap(bboxes, output.shape[-2:])
            binary_mask = expand_to_mask(pred_heatmap)

            # Calculate losses
            seg_loss = dice_loss(output, binary_mask)
            bbox_loss = smooth_l1_loss(extract_bounding_boxes(binary_mask), bboxes)
            total_loss = seg_loss + bbox_loss

            # Backward and optimize
            total_loss.backward()
            optimizer.step()

            total_loss += total_loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")
