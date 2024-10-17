import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import xml.etree.ElementTree as ET
from torchvision import transforms

class NIRDepthDataset(Dataset):
    def __init__(self, nir_dir, depth_dir, xml_dir, transform=None):
        self.nir_dir = nir_dir
        self.depth_dir = depth_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.image_ids = [f.split('.')[0] for f in os.listdir(nir_dir)]

    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        bbox = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            bbox.append([int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                         int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)])
        return torch.tensor(bbox)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        nir_image = Image.open(os.path.join(self.nir_dir, image_id + '.jpg')).convert('RGB')
        depth_image = Image.open(os.path.join(self.depth_dir, image_id + '.jpg')).convert('RGB')
        xml_file = os.path.join(self.xml_dir, image_id + '.xml')
        bbox = self.parse_xml(xml_file)
        if self.transform:
            nir_image = self.transform(nir_image)
            depth_image = self.transform(depth_image)
        return nir_image, depth_image, bbox

# Transformation definition
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
