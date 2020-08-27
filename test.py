from BoxSupDataset.NasaBoxSupDataset import NasaBoxSupDataset
from BoxSupDataset.transforms.denoise import TotalVariation, Bilateral, Wavelet
from BoxSupDataset.utils import show_x_images
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


transformed_testDataset = NasaBoxSupDataset(classfile='classes_bxsp.txt',
                                            rootDir='data/TestBatch',
                                            transform=transforms.Compose([
                                                Wavelet()
                                            ]))

testDataset = NasaBoxSupDataset(classfile='classes_bxsp.txt',
                                rootDir='data/TestBatch'
                                )

print(len(transformed_testDataset))
print(transformed_testDataset[0])

fig = plt.figure()

for i in range(len(transformed_testDataset)):
    sample = transformed_testDataset[i]

    print(i, sample['image'].shape, sample['label'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_x_images(**sample)

    if i == 3:
        plt.show()
        break
for i in range(len(testDataset)):
    sample = testDataset[i]

    print(i, sample['image'].shape, sample['label'].shape)

    ax = plt.subplot(1, 4, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    ax.axis('off')
    show_x_images(**sample)

    if i == 3:
        plt.show()
        break
