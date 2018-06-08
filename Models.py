from custom_layers import *
import network as layer

nc = 3
nz = 512

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        options = {'leaky':True, 'bn':True, 'wn':True, 'pixel':True, 'gdrop':True}
        
        # noise
        self.from_noise = layer.deconv(nz, 512, 4, 1, 0, **options)
        # 4 x 4
        # + 512-vgg-conv
        self.deconv1 = layer.deconv(1024, 512, 4, 2, 1, **options)
        # 8 x 8
        # + 512-vgg-conv
        self.deconv2 = layer.deconv(512, 256, 4, 2, 1, **options)
        # 16 x 16
        # + 256-vgg-conv
        self.deconv3 = layer.deconv(256, 128, 4, 2, 1, **options)
        # 32 x 32
        # + 128-vgg-conv
        self.deconv4 = layer.deconv(128, 64, 4, 2, 1, **options)
        # 64 x 64
        # + 64-vgg-conv
        self.deconv5 = layer.deconv(64, nc, 4, 2, 1, gdrop=options['gdrop'], only=True)
        # 128 x 128
        self.tanh = nn.Tanh()
        
        self.deconvs = [self.deconv1, self.deconv2, self.deconv3, self.deconv4, self.deconv5]
        
    def forward(self, x, features):     
        features = features[::-1]
        x = self.from_noise(x)        
        for i in range(len(self.deconvs)):
            feature, deconv = features[i].detach(), self.deconvs[i]
            if i < 1:
                x = torch.cat([x, feature], 1)
            x = deconv(x)
        x = self.tanh(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        options = {'leaky':True, 'bn':False, 'wn':True, 'pixel':False, 'gdrop':False}
        
        # 128 x 128
        self.from_rgb = layer.conv(nc, 64, 4, 2, 1, **options)
        # 64 x 64
        # + 64-vgg-conv
        self.conv1 = layer.conv(64, 128, 4, 2, 1, **options)
        # 32 x 32
        # + 128-vgg-conv
        self.conv2 = layer.conv(128, 256, 4, 2, 1, **options)
        # 16 x 16
        # + 256-vgg-conv
        self.conv3 = layer.conv(256, 512, 4, 2, 1, **options)
        # 8 x 8
        # + 512-vgg-conv
        self.conv4 = layer.conv(512, 512, 4, 2, 1, **options)
        # 4 x 4
        # + 512-vgg-conv
        self.conv5 = layer.conv(1024, 512, 4, 1, 0, **options)
        # 1 x 1
        self.linear = layer.linear(512, 1, sig=False, wn=options['wn'])
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]
    
    def forward(self, x, features):
        x = self.from_rgb(x)
        for i in range(len(self.convs)):
            feature, conv = features[i].detach(), self.convs[i]
            if i > 3:
                x = torch.cat([x, feature], 1)
            x = conv(x)
        x = self.linear(x)
        
        return x.view(-1, 1)