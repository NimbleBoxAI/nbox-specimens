import torch
from PIL import Image
from pathlib import Path

from torchvision import transforms
from torch.utils.data import Dataset

class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_prompt, tokenizer, size=512, center_crop=False):
        self.tokenizer = tokenizer
        self.size = size
        self.center_crop = center_crop
        
        # Instance images
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError(f"instance_data_root: {instance_data_root} doesn't exist")
        self.instance_images_path = [f for f in list(Path(instance_data_root).iterdir())
                                     if f.suffix.lower() in [   '.png', '.jpg', '.jpeg']]
        self.num_instance_images = len(self.instance_images_path)
        self.instance_prompt = instance_prompt
        self._length = self.num_instance_images
        
        # Image preprocessing
        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        example = {}
        # Weird indexing to accomodate prior_class_preservation
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        example["instance_images"] = self.image_transforms(instance_image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt, padding="do_not_pad", truncation=True, max_length=self.tokenizer.model_max_length
        ).input_ids
        return example

def collate_fn(examples, tokenizer):
    input_ids = [example['instance_prompt_ids'] for example in examples]
    pixel_values = [example['instance_images'] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = tokenizer.pad({'input_ids': input_ids}, 
                              padding='max_length',
                              max_length=tokenizer.model_max_length,
                              return_tensors="pt").input_ids
    batch = {'input_ids': input_ids, 'pixel_values': pixel_values}
    return batch
