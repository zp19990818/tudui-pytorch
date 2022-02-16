# 使用pytorch加载数据

from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

img2 = Image.open('Image\\people\\wangda1.jpg')
# img2.show()

writer = SummaryWriter("venv/logs")

ten_trans = transforms.ToTensor()
tensor_image = ten_trans(img2)

writer.add_image("Tensorimage", tensor_image)
writer.close()