from fastapi import UploadFile, File, FastAPI, HTTPException
import io
import uvicorn
import torch
from torchvision import transforms
import torch.nn as nn
from PIL import Image

class ChekImage(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2)

        )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(16*14*14, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
  def forward(self, x):
    x = self.first(x)
    x = self.second(x)
    return x

transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

check_image_app = FastAPI()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ChekImage()

model.load_state_dict(torch.load('model_fashion_nur.pth', map_location=device))
model.to(device)


@check_image_app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    try:
        image = await file.read()

        if not  image:
            raise HTTPException(status_code=400, detail='Файл кошулган жок')

        img = Image.open(io.BytesIO(image))
        img_tensor = transforms(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(img_tensor)
            pred = y_pred.argmax(dim=1).item()

            classes = {
                0: 'T-shirt/top',
                1: 'Trouser',
                2: 'Pullover',
                3: 'Dress',
                4: 'Coat',
                5: 'Sandal',
                6: 'Shirt',
                7: 'Sneaker',
                8: 'Bag',
                9: 'Ankle boot'
            }

            answer = classes[pred]

            return {'Answer': answer}

        return {'Answer': pred}





    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(check_image_app, host='127.0.0.1', port=8001)

