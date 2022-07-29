from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def show_tensor_images(trainloader, num_images=16):
    '''
    input: trainloader [DataLoader object], num_images [integer]
    output: null
    description: prints multiple tensor images
    '''
    dataiter = iter(trainloader)
    images, _ = dataiter.next()
    unflat_img = images.detach().cpu()
    img_grid = make_grid(unflat_img[:num_images], nrow=4)
    plt.imshow(img_grid.permute(1,2,0).squeeze())
    plt.show()
