import torch
import torch.nn as nn
from torch.nn.modules import pooling
from torch.nn.modules.flatten import Flatten

@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer

################ CBAM ####################################
class Channel_Attention(nn.Module):
    '''Channel Attention in CBAM.
    '''

    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max']):
        '''Param init and architecture building.
        '''

        super(Channel_Attention, self).__init__()
        self.pool_types = pool_types

        self.shared_mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=channel_in, out_features=channel_in//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=channel_in//reduction_ratio, out_features=channel_in)
        )


    def forward(self, x):
        '''Forward Propagation.
        '''

        channel_attentions = []

        for pool_types in self.pool_types:
            if pool_types == 'avg':
                pool_init = nn.AvgPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                avg_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(avg_pool))
            elif pool_types == 'max':
                pool_init = nn.MaxPool2d(kernel_size=(x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                max_pool = pool_init(x)
                channel_attentions.append(self.shared_mlp(max_pool))

        pooling_sums = torch.stack(channel_attentions, dim=0).sum(dim=0)
        scaled = nn.Sigmoid()(pooling_sums).unsqueeze(2).unsqueeze(3).expand_as(x)

        return x * scaled #return the element-wise multiplication between the input and the result.

class ChannelPool(nn.Module):
    '''Merge all the channels in a feature map into two separate channels where the first channel is produced by taking the max values from all channels, while the
       second one is produced by taking the mean from every channel.
    '''
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class Spatial_Attention(nn.Module):
    '''Spatial Attention in CBAM.
    '''

    def __init__(self, kernel_size=7):
        '''Spatial Attention Architecture.
        '''

        super(Spatial_Attention, self).__init__()

        self.compress = ChannelPool()
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, dilation=1, padding=(kernel_size-1)//2, bias=False),
            nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.01, affine=True)
        )


    def forward(self, x):
        '''Forward Propagation.
        '''
        x_compress = self.compress(x)
        x_output = self.spatial_attention(x_compress)
        scaled = nn.Sigmoid()(x_output)
        return x * scaled

class CBAM(nn.Module):
    '''CBAM architecture.
    '''
    def __init__(self, channel_in, reduction_ratio=16, pool_types=['avg', 'max'], spatial=True):
        '''Param init and arch build.
        '''
        super(CBAM, self).__init__()
        self.spatial = spatial

        self.channel_attention = Channel_Attention(channel_in=channel_in, reduction_ratio=reduction_ratio, pool_types=pool_types)

        if self.spatial:
            self.spatial_attention = Spatial_Attention(kernel_size=7)


    def forward(self, x):
        '''Forward Propagation.
        '''
        x_out = self.channel_attention(x)
        if self.spatial:
            x_out = self.spatial_attention(x_out)

        return x_out
##########################################################

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        # self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 3, padding=1)
        self.conv = depthwise_separable_conv(in_channels + cat_channels,out_channels, 3, padding=1 )
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        # self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.conv_1 = depthwise_separable_conv(in_channels,out_channels, 3, padding=1 )
        self.conv_2 = depthwise_separable_conv(out_channels,out_channels, 3, padding=1 )


        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        # self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        # self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_1 = depthwise_separable_conv(in_channels,in_channels, 3, padding=1 )
        self.conv_2 = depthwise_separable_conv(in_channels,out_channels, 3, padding=1 )

        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


# class DenoisingBlock(nn.Module):
#     def __init__(self, in_channels, inner_channels, out_channels):
#         super(DenoisingBlock, self).__init__()
#         # self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
#         # self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
#         # self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
#         # self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

#         self.conv_0 = depthwise_separable_conv(in_channels, inner_channels, 3, padding=1)
#         self.conv_1 = depthwise_separable_conv(in_channels + inner_channels, inner_channels, 3, padding=1)
#         self.conv_2 = depthwise_separable_conv(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
#         self.conv_3 = depthwise_separable_conv(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

#         self.actv_0 = nn.PReLU(inner_channels)
#         self.actv_1 = nn.PReLU(inner_channels)
#         self.actv_2 = nn.PReLU(inner_channels)
#         self.actv_3 = nn.PReLU(out_channels)

#     def forward(self, x):
#         out_0 = self.actv_0(self.conv_0(x))

#         out_0 = torch.cat([x, out_0], 1)
#         out_1 = self.actv_1(self.conv_1(out_0))

#         out_1 = torch.cat([out_0, out_1], 1)
#         out_2 = self.actv_2(self.conv_2(out_1))

#         out_2 = torch.cat([out_1, out_2], 1)
#         out_3 = self.actv_3(self.conv_3(out_2))

#         return out_3 + x
class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()
        # self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        # self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        # self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        # self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

        self.conv_0 = depthwise_separable_conv(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = depthwise_separable_conv(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = depthwise_separable_conv(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = depthwise_separable_conv(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

        self.bn_0 = nn.BatchNorm2d(inner_channels)
        self.bn_1 = nn.BatchNorm2d(inner_channels)
        self.bn_2 = nn.BatchNorm2d(inner_channels)
        self.bn_3 = nn.BatchNorm2d(out_channels)

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

        self.cbam = CBAM(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.bn_0(self.conv_0(x)))

        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.bn_1(self.conv_1(out_0)))

        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.bn_2(self.conv_2(out_1)))

        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.bn_3(self.conv_3(out_2)))

        out_3 = self.cbam(out_3)

        return out_3 + x


class RDUNet(nn.Module):
    r"""
    Residual-Dense U-net for image denoising.
    """
    def __init__(self, **kwargs):
        super().__init__()

        channels = kwargs['channels']
        filters_0 = kwargs['base filters']
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(channels, filters_0)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        # Level 1:
        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        # Level 2:
        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        # Level 3 (Bottleneck)
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)

        # Decoder
        # Level 2:
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)

        # Level 1:
        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)

        # Level 0:
        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)

        self.output_block = OutputBlock(filters_0, channels)

    def forward(self, inputs):
        out_0 = self.input_block(inputs)    # Level 0
        out_0 = self.block_0_0(out_0)
        out_0 = self.block_0_1(out_0)

        out_1 = self.down_0(out_0)          # Level 1
        out_1 = self.block_1_0(out_1)
        out_1 = self.block_1_1(out_1)

        out_2 = self.down_1(out_1)          # Level 2
        out_2 = self.block_2_0(out_2)
        out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)          # Level 3 (Bottleneck)
        out_3 = self.block_3_0(out_3)
        out_3 = self.block_3_1(out_3)

        out_4 = self.up_2([out_3, out_2])   # Level 2
        out_4 = self.block_2_2(out_4)
        out_4 = self.block_2_3(out_4)

        out_5 = self.up_1([out_4, out_1])   # Level 1
        out_5 = self.block_1_2(out_5)
        out_5 = self.block_1_3(out_5)

        out_6 = self.up_0([out_5, out_0])   # Level 0
        out_6 = self.block_0_2(out_6)
        out_6 = self.block_0_3(out_6)

        return self.output_block(out_6) + inputs
if __name__ == '__main__':
    import yaml
    from utils import set_seed
    from ptflops import get_model_complexity_info
    with open('config.yaml', 'r') as stream:                # Load YAML configuration file.
        config = yaml.safe_load(stream)

    model_params = config['model']
    train_params = config['train']
    val_params = config['val']

    # Defining model:
    set_seed(0)
    # model = RDUNet(**model_params)

    # print('Model summary:')
    # test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    # with torch.no_grad():
    #     macs, params = get_model_complexity_info(model, test_shape, as_strings=True,
    #                                              print_per_layer_stat=False, verbose=False)
    #     print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    #     print('{:<30}  {:<8}'.format('Number of parameters: ', params))



    ###########visual model #################
    # from torchviz import make_dot
    # model = RDUNet(**model_params)

    # # Create a dummy input tensor
    # test_shape = (model_params['channels'], train_params['patch size'], train_params['patch size'])
    # print(test_shape)
    # # Get the output of the model
    # x = torch.randn(1, 3, 64, 64)
    
    # # out = model(test_shape)
    # # Visualize the model graph
    # dot = make_dot(model(x), params=dict(model.named_parameters()),show_attrs=True)
    # dot.render('model', format='png')


    ############## visual by torch #############
    # from torch.utils.tensorboard import SummaryWriter

    # writer = SummaryWriter("torchlogs/")
    # model = RDUNet(**model_params)
    # X = torch.randn(1, 3, 64, 64)
    # writer.add_graph(model, X)
    # writer.close()