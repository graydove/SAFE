import numpy as np
import random
import torch
from PIL import Image
from models.resnet import resnet50
from torchvision import transforms


transform = transforms.Compose([transforms.CenterCrop([256,256]),transforms.ToTensor()])

def imagedetection(image_path, model, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).to(device)
    image = image.unsqueeze(0) 

    with torch.no_grad():
        output = model(image)

    predictions = torch.softmax(output, axis=1)[:, 1]
    y_pred = predictions.cpu().numpy()
    return y_pred

if __name__ == '__main__':
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda')
    model = resnet50(num_classes=2)

    # load checkpoint
    checkpoint = torch.load('./checkpoint/checkpoint-best.pth')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()

    image_path = "0_adm_153.PNG"
    predictions = imagedetection(image_path, model, device)
    print(predictions)