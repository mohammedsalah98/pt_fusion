import torch
import torch.nn as nn

#----------------------- Attention 2D UNet Multi-Modal -----------------------#
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class att_UNetMultiModal_MultiClass(nn.Module):
    def __init__(self, in_channels_pca, in_channels_tsr, num_classes):
        super(att_UNetMultiModal_MultiClass, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pca_enc1 = self.residual_conv(in_channels_pca, 64)
        self.pca_enc2 = self.residual_conv(64, 128)
        self.pca_enc3 = self.residual_conv(128, 256)
        self.pca_enc4 = self.residual_conv(256, 512)
        self.pca_enc5 = self.residual_conv(512, 1024)

        self.tsr_enc1 = self.residual_conv(in_channels_tsr, 64)
        self.tsr_enc2 = self.residual_conv(64, 128)
        self.tsr_enc3 = self.residual_conv(128, 256)
        self.tsr_enc4 = self.residual_conv(256, 512)
        self.tsr_enc5 = self.residual_conv(512, 1024)

        self.pca_bottleneck = self.residual_conv(1024, 2048)
        self.tsr_bottleneck = self.residual_conv(1024, 2048)

        self.conv1x1_1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv1x1_bottleneck = nn.Conv2d(2048, 2048, kernel_size=1)

        self.up_conv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.att5 = AttentionBlock(F_g=1024, F_l=1024, F_int=512)
        self.dec_conv5 = self.residual_conv(2048, 1024)

        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec_conv4 = self.residual_conv(1024, 512)

        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec_conv3 = self.residual_conv(512, 256)

        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec_conv2 = self.residual_conv(256, 128)

        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec_conv1 = self.residual_conv(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    class ResidualConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            identity = self.shortcut(x)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity
            out = self.relu(out)
            return out

    def residual_conv(self, in_channels, out_channels):
        return self.ResidualConvBlock(in_channels, out_channels)

    def forward(self, x_pca, x_tsr):
        pca_enc1 = self.pca_enc1(x_pca)
        pca_enc2 = self.pca_enc2(self.pool(pca_enc1))
        pca_enc3 = self.pca_enc3(self.pool(pca_enc2))
        pca_enc4 = self.pca_enc4(self.pool(pca_enc3))
        pca_enc5 = self.pca_enc5(self.pool(pca_enc4))

        tsr_enc1 = self.tsr_enc1(x_tsr)
        tsr_enc2 = self.tsr_enc2(self.pool(tsr_enc1))
        tsr_enc3 = self.tsr_enc3(self.pool(tsr_enc2))
        tsr_enc4 = self.tsr_enc4(self.pool(tsr_enc3))
        tsr_enc5 = self.tsr_enc5(self.pool(tsr_enc4))

        pca_bottleneck = self.pca_bottleneck(self.pool(pca_enc5))
        tsr_bottleneck = self.tsr_bottleneck(self.pool(tsr_enc5))

        tsr_weight_bottleneck = torch.sigmoid(self.conv1x1_bottleneck(tsr_bottleneck))
        bottleneck = (tsr_weight_bottleneck * tsr_bottleneck) + ((1 - tsr_weight_bottleneck) * pca_bottleneck)

        tsr_weight_5 = torch.sigmoid(self.conv1x1_5(tsr_enc5))
        fusion_5 = (tsr_weight_5 * tsr_enc5) + ((1 - tsr_weight_5) * pca_enc5)

        tsr_weight_4 = torch.sigmoid(self.conv1x1_4(tsr_enc4))
        fusion_4 = (tsr_weight_4 * tsr_enc4) + ((1 - tsr_weight_4) * pca_enc4)

        tsr_weight_3 = torch.sigmoid(self.conv1x1_3(tsr_enc3))
        fusion_3 = (tsr_weight_3 * tsr_enc3) + ((1 - tsr_weight_3) * pca_enc3)

        tsr_weight_2 = torch.sigmoid(self.conv1x1_2(tsr_enc2))
        fusion_2 = (tsr_weight_2 * tsr_enc2) + ((1 - tsr_weight_2) * pca_enc2)

        tsr_weight_1 = torch.sigmoid(self.conv1x1_1(tsr_enc1))
        fusion_1 = (tsr_weight_1 * tsr_enc1) + ((1 - tsr_weight_1) * pca_enc1)

        dec5 = self.up_conv5(bottleneck)
        fusion_5 = self.att5(dec5, fusion_5)
        dec5 = torch.cat((fusion_5, dec5), dim=1)
        dec5 = self.dec_conv5(dec5)

        dec4 = self.up_conv4(dec5)
        fusion_4 = self.att4(dec4, fusion_4)
        dec4 = torch.cat((fusion_4, dec4), dim=1)
        dec4 = self.dec_conv4(dec4)

        dec3 = self.up_conv3(dec4)
        fusion_3 = self.att3(dec3, fusion_3)
        dec3 = torch.cat((fusion_3, dec3), dim=1)
        dec3 = self.dec_conv3(dec3)

        dec2 = self.up_conv2(dec3)
        fusion_2 = self.att2(dec2, fusion_2)
        dec2 = torch.cat((fusion_2, dec2), dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.up_conv1(dec2)
        fusion_1 = self.att1(dec1, fusion_1)
        dec1 = torch.cat((fusion_1, dec1), dim=1)
        dec1 = self.dec_conv1(dec1)

        output = self.final_conv(dec1)

        return output

#----------------------- Attention 2D UNet Multi-Modal Depth -----------------------#
class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class att_UNetMultiModal_depth_MultiClass(nn.Module):
    def __init__(self, in_channels_pca, in_channels_tsr, num_classes):
        super(att_UNetMultiModal_depth_MultiClass, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.pca_enc1 = self.residual_conv(in_channels_pca, 64)
        self.pca_enc2 = self.residual_conv(64, 128)
        self.pca_enc3 = self.residual_conv(128, 256)
        self.pca_enc4 = self.residual_conv(256, 512)
        self.pca_enc5 = self.residual_conv(512, 1024)

        self.tsr_enc1 = self.residual_conv(in_channels_tsr, 64)
        self.tsr_enc2 = self.residual_conv(64, 128)
        self.tsr_enc3 = self.residual_conv(128, 256)
        self.tsr_enc4 = self.residual_conv(256, 512)
        self.tsr_enc5 = self.residual_conv(512, 1024)

        self.pca_bottleneck = self.residual_conv(1024, 2048)
        self.tsr_bottleneck = self.residual_conv(1024, 2048)

        self.conv1x1_1 = nn.Conv2d(64, 64, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(128, 128, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(256, 256, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(1024, 1024, kernel_size=1)
        self.conv1x1_bottleneck = nn.Conv2d(2048, 2048, kernel_size=1)

        self.up_conv5 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.att5 = AttentionBlock(F_g=1024, F_l=1024, F_int=512)
        self.dec_conv5 = self.residual_conv(2048, 1024)

        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec_conv4 = self.residual_conv(1024, 512)

        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec_conv3 = self.residual_conv(512, 256)

        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec_conv2 = self.residual_conv(256, 128)

        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec_conv1 = self.residual_conv(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

        self.final_conv_depth = nn.Conv2d(64, 1, kernel_size=1)

    class ResidualConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
            self.bn2 = nn.BatchNorm2d(out_channels)

            self.shortcut = nn.Sequential()
            if in_channels != out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                    nn.BatchNorm2d(out_channels)
                )

        def forward(self, x):
            identity = self.shortcut(x)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += identity
            out = self.relu(out)
            return out

    def residual_conv(self, in_channels, out_channels):
        return self.ResidualConvBlock(in_channels, out_channels)

    def forward(self, x_pca, x_tsr):
        pca_enc1 = self.pca_enc1(x_pca)
        pca_enc2 = self.pca_enc2(self.pool(pca_enc1))
        pca_enc3 = self.pca_enc3(self.pool(pca_enc2))
        pca_enc4 = self.pca_enc4(self.pool(pca_enc3))
        pca_enc5 = self.pca_enc5(self.pool(pca_enc4))

        tsr_enc1 = self.tsr_enc1(x_tsr)
        tsr_enc2 = self.tsr_enc2(self.pool(tsr_enc1))
        tsr_enc3 = self.tsr_enc3(self.pool(tsr_enc2))
        tsr_enc4 = self.tsr_enc4(self.pool(tsr_enc3))
        tsr_enc5 = self.tsr_enc5(self.pool(tsr_enc4))

        pca_bottleneck = self.pca_bottleneck(self.pool(pca_enc5))
        tsr_bottleneck = self.tsr_bottleneck(self.pool(tsr_enc5))

        tsr_weight_bottleneck = torch.sigmoid(self.conv1x1_bottleneck(tsr_bottleneck))
        bottleneck = (tsr_weight_bottleneck * tsr_bottleneck) + ((1 - tsr_weight_bottleneck) * pca_bottleneck)

        tsr_weight_5 = torch.sigmoid(self.conv1x1_5(tsr_enc5))
        fusion_5 = (tsr_weight_5 * tsr_enc5) + ((1 - tsr_weight_5) * pca_enc5)

        tsr_weight_4 = torch.sigmoid(self.conv1x1_4(tsr_enc4))
        fusion_4 = (tsr_weight_4 * tsr_enc4) + ((1 - tsr_weight_4) * pca_enc4)

        tsr_weight_3 = torch.sigmoid(self.conv1x1_3(tsr_enc3))
        fusion_3 = (tsr_weight_3 * tsr_enc3) + ((1 - tsr_weight_3) * pca_enc3)

        tsr_weight_2 = torch.sigmoid(self.conv1x1_2(tsr_enc2))
        fusion_2 = (tsr_weight_2 * tsr_enc2) + ((1 - tsr_weight_2) * pca_enc2)

        tsr_weight_1 = torch.sigmoid(self.conv1x1_1(tsr_enc1))
        fusion_1 = (tsr_weight_1 * tsr_enc1) + ((1 - tsr_weight_1) * pca_enc1)

        dec5 = self.up_conv5(bottleneck)
        fusion_5 = self.att5(dec5, fusion_5)
        dec5 = torch.cat((fusion_5, dec5), dim=1)
        dec5 = self.dec_conv5(dec5)

        dec4 = self.up_conv4(dec5)
        fusion_4 = self.att4(dec4, fusion_4)
        dec4 = torch.cat((fusion_4, dec4), dim=1)
        dec4 = self.dec_conv4(dec4)

        dec3 = self.up_conv3(dec4)
        fusion_3 = self.att3(dec3, fusion_3)
        dec3 = torch.cat((fusion_3, dec3), dim=1)
        dec3 = self.dec_conv3(dec3)

        dec2 = self.up_conv2(dec3)
        fusion_2 = self.att2(dec2, fusion_2)
        dec2 = torch.cat((fusion_2, dec2), dim=1)
        dec2 = self.dec_conv2(dec2)

        dec1 = self.up_conv1(dec2)
        fusion_1 = self.att1(dec1, fusion_1)
        dec1 = torch.cat((fusion_1, dec1), dim=1)
        dec1 = self.dec_conv1(dec1)

        output = self.final_conv(dec1)
        output_depth = self.final_conv_depth(dec1)

        return output, output_depth