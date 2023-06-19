from torch.optim.lr_scheduler import *
import functools
from basicsr.utils.registry import ARCH_REGISTRY
from ESRGAN_master.HRTF.config import *
from ESRGAN_master.HRTF.hrtfdata.transforms.hrirs import SphericalHarmonicsTransform
from ESRGAN_master.HRTF.model.util import *
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

path = "/rds/general/user/jl2622/home/Upsample_GAN/ESRGAN_master/models/model"
GPU = True # Choose whether to use GPU
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(f'Using {device}')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
config = Config("ari-upscale-4", using_hpc=True)
data_dir = config.raw_hrtf_dir / config.dataset
imp = importlib.import_module('hrtfdata.full')
load_function = getattr(imp, config.dataset)
# #### Data loading

# In[4]:
def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


# class ResidualDenseBlock_5C(nn.Module):
#     def __init__(self, nf=64, gc=32, bias=True):
#         super(ResidualDenseBlock_5C, self).__init__()
#         # gc: growth channel, i.e. intermediate channels
#         self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
#         self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
#         self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
#         self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
#         self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
#
#         # initialization
#         # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)
#
#     def forward(self, x):
#         x1 = self.lrelu(self.conv1(x))
#         x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
#         x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
#         x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
#         x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
#         return x5 * 0.2 + x
class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super(ResidualDenseBlock_5C, self).__init__()
        # gc: growth channel, i.e. intermediate channels
        self.conv1 = nn.Conv1d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv1d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv1d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv1d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv1d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # mutil.initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock_5C(nf, gc)
        self.RDB2 = ResidualDenseBlock_5C(nf, gc)
        self.RDB3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x

class Upsample(nn.Module):
    def __init__(self, scale_factor):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

class Downsample(nn.Module):
    def __init__(self, output_size):
        super(Downsample, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return F.adaptive_avg_pool1d(x, self.output_size)

class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32):
        super(RRDBNet, self).__init__()
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv1d(in_nc, nf, 3, 1, 1, bias=True)
        # self.conv_first = nn.Conv1d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.RRDB_trunk = make_layer(RRDB_block_f, nb)
        self.trunk_conv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv1d(nf, nf, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Upsampling and Downsampling layers
        self.up = Upsample(scale_factor=3)  # 784*3 = 2352
        self.down = Downsample(output_size=841)  # downsample to 841

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv4(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        # Apply Upsampling and Downsampling
        out = self.up(out)
        out = self.down(out)

        return out


@ARCH_REGISTRY.register()
class VGGStyleDiscriminator1D(nn.Module):
    def __init__(self, num_in_ch, num_feat, input_size=841):
        super(VGGStyleDiscriminator1D, self).__init__()
        self.input_size = input_size

        self.conv0_0 = nn.Conv1d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv1d(num_feat, num_feat, 3, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm1d(num_feat, affine=True)

        self.conv1_0 = nn.Conv1d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm1d(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv1d(num_feat * 2, num_feat * 2, 3, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm1d(num_feat * 2, affine=True)
        # 1664, 100
        self.linear1 = nn.Linear(27008, 100)  # Adjusted input dimension
        self.linear2 = nn.Linear(100, 1)

            # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == self.input_size, (f'Input size must be identical to input_size, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main():
    # Set batch size
    batch_size = 4
    # Apply data loader
    config = Config("ari-upscale-4", using_hpc=False)
    loader_train, loader_test= load_hrtf(config)
    print(len(loader_train), len(loader_test))

    num_epochs = 200
    learning_rate = 2e-4
    latent_vector_size = 100
    use_weights_init = True

    model_G = RRDBNet(256, 128, 256, 23, gc=32).to(device)
    model_G = model_G.float()
    if use_weights_init:
        model_G.apply(weights_init)
    params_G = sum(p.numel() for p in model_G.parameters() if p.requires_grad)
    print("Total number of parameters in Generator is: {}".format(params_G))
    print(model_G)
    print('\n')
    # [256, 64]
    model_D = VGGStyleDiscriminator1D(256, 64).to(device)
    # double
    model_D = model_D.float()
    if use_weights_init:
        model_D.apply(weights_init)
    params_D = sum(p.numel() for p in model_D.parameters() if p.requires_grad)
    print("Total number of parameters in Discriminator is: {}".format(params_D))
    print(model_D)
    print('\n')

    print("Total number of parameters is: {}".format(params_G + params_D))

    def loss_function(out, label):
        criterion = nn.BCEWithLogitsLoss()
        loss = criterion(out, label)
        return loss

    Disc_lr = 2e-4
    beta1 = 0.5
    Gene_lr = 1e-4
    optimizerD = torch.optim.Adam(model_D.parameters(), lr=Disc_lr, betas=(beta1, 0.999))
    optimizerG = torch.optim.Adam(model_G.parameters(), lr=Gene_lr, betas=(beta1, 0.999))

    # Additional input variables should be defined here
    False_img_label = 0
    # Mode Collapse can be avoid by initalise Real_label to 0.9 instead of 1
    True_img_label = 0.9

    # Start training loop
    train_losses_G = []
    train_losses_D = []
    content_losses = []
    schedulerD = StepLR(optimizerD, step_size=20, gamma=0.93)

    for epoch in range(num_epochs):
        train_loss_D = 0
        train_loss_G = 0
        loader_train.reset()
        batch_data = loader_train.next()
        print("Epoch:", epoch)
        counter = 0
        while batch_data is not None:
                model_D.zero_grad()

                lr_coefficient = batch_data["lr_coefficient"].to(device=device)
                lr_coefficient = lr_coefficient.float()
                hr_coefficient = batch_data["hr_coefficient"].to(device=device)
                hr_coefficient = hr_coefficient.float()
                hrir = batch_data["hrir"].to(device=device, memory_format=torch.contiguous_format,
                                     non_blocking=True, dtype=torch.float)
                masks = batch_data["mask"].to(device=device, memory_format=torch.contiguous_format,
                                      non_blocking=True, dtype=torch.float)
                size_batch = lr_coefficient.size(0)
                img_label = torch.full((size_batch,), True_img_label, device=device, dtype=torch.float)
                img_label = img_label.reshape(1, 1)
                real_D_output = model_D(hr_coefficient)
                real_D_error = loss_function(real_D_output, img_label)
                real_D_error.backward()
                # Output true mean discriminator result
                D_x = real_D_output.mean().item()

                # Start to train discriminator with fake image
                # image_noise = torch.randn(size_batch, latent_vector_size, 1, 1, device=device)
                # image_noise = image_noise.view(1, -1)
                # [1, 128, 49]
                img_fake = model_G(lr_coefficient)
                img_label.fill_(False_img_label)
                real_D_output = model_D(img_fake.detach())
                fake_D_error = loss_function(real_D_output, img_label)
                fake_D_error.backward()
                D_G_z1 = real_D_output.mean().item()
                errD = real_D_error + fake_D_error
                train_loss_D = train_loss_D + errD.item()
                optimizerD.step()

                # Start to maximizing log(D(G(z))) to update G network
                model_G.zero_grad()
                content_criterion = sd_ild_loss
                recon_coef_list = []
                ds = load_function(data_dir, feature_spec={
                    'hrirs': {'samplerate': config.hrir_samplerate, 'side': 'left', 'domain': 'magnitude'}})
                num_row_angles = len(ds.row_angles)
                num_col_angles = len(ds.column_angles)
                num_radii = len(ds.radii)
                # HRTF Frequency(one ear 128, if concate two ears, it should be 256)
                nbins = config.nbins_hrtf
                # mean and std for ILD and SD, which are used for normalization
                # computed based on average ILD and SD for training data, when comparing each individual
                # to every other individual in the training data
                sd_mean = 7.387559253346883
                sd_std = 0.577364154400081
                ild_mean = 3.6508303231127868
                ild_std = 0.5261339271318863
                if config.merge_flag:
                    nbins = config.nbins_hrtf * 2
                for i in range(masks.size(0)):
                    SHT = SphericalHarmonicsTransform(28, ds.row_angles, ds.column_angles, ds.radii,
                                                      masks[i].detach().cpu().numpy().astype(bool))
                    harmonics = torch.from_numpy(SHT.get_harmonics()).float().to(device)
                    # recon_hrir = SHT.inverse(img_fake[i].T.detach().cpu().numpy())  # Compute the inverse
                    # recon_hrir = harmonics @ img_fake[i].T.detach().cpu().numpy()
                    recon_hrir = (harmonics @ img_fake[i].T).detach().cpu().numpy()
                    recon_hrir = torch.from_numpy(recon_hrir).type(torch.float32)
                    # recon_hrir = recon_hrir.type(torch.float32)
                    # recon_hrir_tensor = torch.from_numpy(recon_hrir.T).reshape(nbins, num_radii, num_row_angles, num_col_angles)
                    recon_hrir_tensor = recon_hrir.transpose(0, 1).reshape(nbins, num_radii, num_row_angles,
                                                                           num_col_angles)
                    recon_coef_list.append(recon_hrir_tensor)
                counter += 1
                recons = torch.stack(recon_coef_list).to(device)
                unweighted_content_loss = content_criterion(config, recons, hrir, sd_mean, sd_std, ild_mean, ild_std)
                content_loss = config.content_weight * unweighted_content_loss
                img_label.fill_(True_img_label)
                img_fake_result = model_D(img_fake)
                errG = loss_function(img_fake_result, img_label)
                errG.backward()
                D_G_z2 = img_fake_result.mean().item()
                train_loss_G = train_loss_G + errG.item()
                print(train_loss_G)
                total_content_loss = errG.item() + content_loss
                print(total_content_loss)
                optimizerG.step()

                #######################################################################
                #                       ** END OF YOUR CODE **
                #######################################################################
                # Logging
                if counter % 50 == 0:
                    print(f"Batch:{counter}/{len(loader_train)}")
                    # tepoch.set_description(f"Epoch {epoch}")
                    # tepoch.set_postfix(D_G_z=f"{D_G_z1:.3f}/{D_G_z2:.3f}", D_x=D_x,
                    #                    Loss_D=errD.item(), Loss_G=errG.item())
                if counter == 0:
                    torch.save(model_G.state_dict(), f'{path}/ GAN_G_model.pth')
                    torch.save(model_D.state_dict(), f'{path}/ GAN_D_model.pth')
                i += 1
                batch_data = loader_train.next()
        # torch.save(model_G.state_dict(), f'{path}/ GAN_G_model.pth')
        # torch.save(model_D.state_dict(), f'{path}/ GAN_D_model.pth')
        schedulerD.step()
        # if epoch == 0:
        #     save_image(denorm(lr_coefficient.cpu()).float(), content_path / 'CW_GAN/real_samples.png')
        # with torch.no_grad():
        #     fake = model_G(fixed_noise)
        #     img_path = "CW_GAN/fake_samples_epoch %03d.png" % epoch
        #     save_image(denorm(fake.cpu()).float(), content_path / img_path)
            # save_image(denorm(fake.cpu()).float(), content_path/'CW_GAN/fake_samples_epoch_%03d.png' % epoch)
        train_losses_D.append(train_loss_D / len(loader_train))
        train_losses_G.append(train_loss_G / len(loader_train))
        content_losses.append(total_content_loss / len(loader_train))

        batch_data = loader_train.next()
        print("Finish")
    # save  models
    #  if your discriminator/generator are conditional you'll want to change the inputs here
    # torch.jit.save(torch.jit.trace(model_G, (fixed_noise)), content_path / 'CW_GAN/GAN_G_model.pth')
    # torch.jit.save(torch.jit.trace(model_D, (fake)), content_path / 'CW_GAN/GAN_D_model.pth')

    # ### Generator samples
    #
    # input_noise = torch.randn(100, latent_vector_size, 1, 1, device=device)
    # model_G = torch.jit.load(content_path/'CW_GAN/GAN_G_model.pth')
    # with torch.no_grad():
    #     # visualize the generated images
    #     generated = model_G(input_noise).cpu()
    #     generated = make_grid(denorm(generated)[:100], nrow=10, padding=2, normalize=False,
    #                         range=None, scale_each=False, pad_value=0)
    #     plt.figure(figsize=(10,10))
    #     save_image(generated, content_path/'CW_GAN/Teaching_final.png')
    #     show(generated) # note these are now class conditional images columns rep classes 1-10
    #
    # it = iter(loader_test)
    # sample_inputs, _ = next(it)
    # fixed_input = sample_inputs[0:64, :, :, :]
    # # visualize the original images of the last batch of the test set for comparison
    # img = make_grid(denorm(fixed_input), nrow=8, padding=2, normalize=False,
    #                 range=None, scale_each=False, pad_value=0)
    # plt.figure(figsize=(10,10))
    # show(img)
    #
    # ANSWER FOR PART 2.2 IN THIS CELL*

    # Plot loss curves
    figure, axis = plt.subplots(1,2,figsize=(16,5))
    figure.suptitle(f'the losses curves for the discriminator D and the generator G as the training progresses')

    axis[0].set_xlabel('Number of Epochs')
    axis[0].set_ylabel('Model Loss')
    axis[0].plot(train_losses_G, 'b', label='Generator')
    leg = axis[0].legend(loc='upper right')

    axis[1].set_xlabel('Number of Epochs')
    axis[1].set_ylabel('Model Loss')
    axis[1].plot(train_losses_D, 'r', label='Discriminator')
    leg = axis[1].legend(loc='upper right')
    # Save the figure before calling plt.show()
    plt.savefig('loss_curve.png', dpi=300)
    plt.show()

    # Plot content_loss curve
    plt.figure(figsize=(8, 5))
    plt.title('The losses curve for the Generator as the training progresses')

    plt.xlabel('Number of Epochs')
    plt.ylabel('Model Loss')
    plt.plot(content_loss, 'b', label='Generator')
    plt.legend(loc='upper right')
    # Save the figure before calling plt.show()
    plt.savefig('content_loss_curve.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    main()





