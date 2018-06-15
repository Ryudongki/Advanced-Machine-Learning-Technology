from custom_layers import *
import network as layer

nc = 3
nz = 100
feature_size = 512

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        options = {'leaky':True, 'bn':True, 'wn':False, 'pixel':True, 'gdrop':True}
        
#         self.from_feature = layer.deconv(feature_size, 256, 4, 1, 0, **options)
        # noise
        self.from_noise = layer.deconv(nz + feature_size, 512, 4, 1, 0, **options)
        # 4 x 4
        # + 256-feature-conv
        self.deconv1 = layer.deconv(512, 256, 4, 2, 1, **options)
        # 8 x 8
        self.deconv2 = layer.deconv(256, 128, 4, 2, 1, **options)
        # 16 x 16
        self.deconv3 = layer.deconv(128, 64, 4, 2, 1, **options)
        # 32 x 32
        self.deconv4 = layer.deconv(64, nc, 4, 2, 1, gdrop=options['gdrop'], only=True)
        # 64 x 64
        self.tanh = nn.Tanh()
        
        self.deconvs = [self.deconv1, self.deconv2, self.deconv3, self.deconv4]
        
    def forward(self, x, feature):
        x = torch.cat([x, feature.view(-1, 512, 1, 1)], 1)
        x = self.from_noise(x)
#         feature = self.from_feature(feature.view(-1, 512, 1, 1))        
        for deconv in self.deconvs:
            x = deconv(x)
        x = self.tanh(x)

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        options = {'leaky':True, 'bn':True, 'wn':False, 'pixel':False, 'gdrop':True}
        
        self.from_feature = layer.linear(feature_size, 64 * 64, leaky=options['leaky'], wn=options['wn'])
        self.feature_conv = layer.conv(1, 64, 4, 2, 1, **options)
        # 64 x 64
        self.from_rgb = layer.conv(nc, 64, 4, 2, 1, leaky=True, bn=False, gdrop=True)
        # 32 x 32
        self.conv1 = layer.conv(128, 128, 4, 2, 1, **options)
        # 16 x 16
        self.conv2 = layer.conv(128, 256, 4, 2, 1, **options)
        # 8 x 8
        self.conv3 = layer.conv(256, 512, 4, 2, 1, **options)
        # 4 x 4
        self.conv4 = nn.Sequential(
            minibatch_std_concat_layer(),
            layer.conv(513, 1, 4, 1, 0,  gdrop=options['gdrop'], only=True)
        )
        # 1 x 1
#         self.linear = layer.linear(512, 1, sig=False, wn=options['wn'])
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4]
    
    def forward(self, x, feature):
        x = self.from_rgb(x)
        feature = self.from_feature(feature).view(-1, 1, 64, 64)
        feature = self.feature_conv(feature)
        x = torch.cat([x, feature], 1)
        
        for conv in self.convs:
            x = conv(x)
#         x = self.linear(x)
        
        return x.view(-1, 1)