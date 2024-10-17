import os
import torch
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

class NIRDepthDataset(Dataset):
    def __init__(self, nir_dir, depth_dir, xml_dir, transform=None):
        self.nir_dir = nir_dir
        self.depth_dir = depth_dir
        self.xml_dir = xml_dir
        self.transform = transform
        self.image_ids = [f.split('.')[0] for f in os.listdir(nir_dir)]
        
    def __len__(self):
        return len(self.image_ids)
    
    def parse_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        bboxes = []
        for obj in root.findall('object'):
            bndbox = obj.find('bndbox')
            bbox = [
                int(bndbox.find('xmin').text),
                int(bndbox.find('ymin').text),
                int(bndbox.find('xmax').text),
                int(bndbox.find('ymax').text),
            ]
            bboxes.append(bbox)

        return torch.tensor(bboxes)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        nir_image = Image.open(os.path.join(self.nir_dir, image_id + '.jpg')).convert('RGB')
        depth_image = Image.open(os.path.join(self.depth_dir, image_id + '.jpg')).convert('RGB')

        xml_file = os.path.join(self.xml_dir, image_id + '.xml')
        bboxes = self.parse_xml(xml_file)

        if self.transform:
            nir_image = self.transform(nir_image)
            depth_image = self.transform(depth_image)

        return nir_image, depth_image, bboxes
