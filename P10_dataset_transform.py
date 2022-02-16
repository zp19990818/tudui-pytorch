# P14 torchvision中的数据集使用

import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set = torchvision.datasets.CIFAR10(root="./datasets", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./datasets", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# img, target = test_set[0]
# print(img)
# print(target)
# img.show()

writer = SummaryWriter("P10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)


writer.close()
