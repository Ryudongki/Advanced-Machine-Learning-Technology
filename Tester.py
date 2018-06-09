# TODO LIST FROM GANHACKS
# https://github.com/soumith/ganhacks
# ✓ Normalize the inputs
# ✓ A modified loss function
# ✓ Use a spherical Z
# ✓ BatchNorm
# Aviod sparse gradients
# ✓ Use soft and noisy labels
# ✓ DCGAN
# Use stability tricks from RL
# ✓ Use SGD for discriminator ADAM for generator
# ✓ Add noise to inputs
# Batch Discrimination (for diversity)
# ✓ Use dropouts in G in both train and test phase


# In[2]:


import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data
import torchvision.datasets as dset
from Folder import ImageFeatureFolder
import net_sphere
nc = 3
nz = 512
lr     = 0.0002
beta1  = 0.0   
beta2  = 0.99     
imageSize = 64
batchSize = 64

outf = "./celeba_test/"
des_dir = "./celeba_testimage/"

dataset = dset.ImageFolder(root=des_dir,
                        transform=transforms.Compose([
                            transforms.CenterCrop(178),
                            transforms.Resize(imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size= batchSize,
                                         shuffle=False)


# In[3]:


from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.utils as vutils
from Models import Generator, Discriminator
from VGG import VGG16Feature


# In[4]:


netG = Generator()
feature_net = net_sphere.sphere20a(feature=True)

criterion = nn.MSELoss()

input = torch.FloatTensor(batchSize, 3, imageSize,imageSize)
noise = torch.FloatTensor(batchSize, nz, 1, 1)
feature = torch.FloatTensor(batchSize, 512, 7, 7)
fixed_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)

label_real = torch.FloatTensor(batchSize)
label_real_smooth = torch.FloatTensor(batchSize)
label_fake = torch.FloatTensor(batchSize)

netG.cuda()
criterion.cuda()
feature_net.cuda()
feature_net.load_state_dict(torch.load('./models/sphere20a_20171020.pth'))
feature_net.eval()
input, feature, noise = input.cuda(), feature.cuda(), noise.cuda()
label_real, label_real_smooth, label_fake = label_real.cuda(), label_real_smooth.cuda(), label_fake.cuda()
fixed_noise = fixed_noise.cuda()

label_real.resize_(batchSize, 1).fill_(1)
label_fake.resize_(batchSize, 1).fill_(0)
label_real_smooth.resize_(batchSize, 1).fill_(0.9)
label_real = Variable(label_real)
label_fake = Variable(label_fake)
label_real_smooth = Variable(label_real_smooth)
print()


# In[5]:


netG.load_state_dict(torch.load('./result/netG_epoch_011.pth'))


# In[6]:

feature_input_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.CenterCrop(128), 
    transforms.Resize(112), 
    transforms.CenterCrop((112, 96)),
    transforms.ToTensor()
])

def transform_batch(batch, transform=feature_input_transform):
    ret_list = []
    for i in range(batch.shape[0]):
        ret_list.append(transform(batch[i]).view(1, 3, 112, 96))
    return torch.cat(ret_list, 0)

with torch.no_grad():
    for i, data in enumerate(dataloader, 0):
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        real = real_cpu.cuda()
        input.resize_as_(real).copy_(real)
        inputv = Variable(input)
        
        vutils.save_image(input.data,
                '%s/input.png' % outf,
                normalize=True)
        feature_input_real = transform_batch(real_cpu).cuda()

        vutils.save_image(feature_input_real.data, '%s/feature.png' % outf, normalize=True)
        feature_inputv_real = Variable(feature_input_real)
        feature_real = feature_net(feature_inputv_real)
        feature_real = Variable(feature_real.data[7].repeat(64, 1, 1, 1))
        
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        
        fake = netG(noisev, feature_real.detach())
        
        vutils.save_image(fake.data,
                '%s/output.png' % outf,
                normalize=True)
        break