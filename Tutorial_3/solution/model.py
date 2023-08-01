import torch
import torch.nn as nn

""" 
Information about architecture config:
Tuple is structured by (kernel_size, filters, stride, padding) 
"M" is simply maxpooling with stride 2x2 and kernel 2x2
List is structured by tuples and lastly int with number of repeats
"""

architecture_config = [
    (7, 64, 2, 3), # For example this is a 7x7 conv layer with 64 kernels using a stride of 2 and padding of 3
    "M", # For example this is a maxpool layer
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4], # For example this is a (1x1 conv layer with 256 kernels stride 1 padding 0 followed by a 3x3 conv layer with 512 kernels stride 1 padding 1) x 4 
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]

"""
This is a basic conv layer that can be used as a building block for our actual model.
Basically helps us reduce repetition of code.
"""
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs): #kwargs if we do end up having some keyword arguments in future
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs) # We don't add any bias terms
        self.batchnorm = nn.BatchNorm2d(out_channels) # BatchNorm is recommended as we will be batching
        self.leakyrelu = nn.LeakyReLU(0.1) 

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))
    
"""
This is our main model.
"""
class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs): # dealing with RGB images so we have a 3 channel input
        super(Yolov1, self).__init__()
        self.architecture = architecture_config # this allows us to change up the architecture by changing the config, which we could actually move to another file (e.g config.py)
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture) # generate our CNN network based on network architecture -> Paper refers to this CNN as "Darknet"
        self.fcs = self._create_fcs(**kwargs) # generate our FCN to perform classification

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1)) # remember to flatten before we pass it to our FCN

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple: # If its a tuple, its simply a CNN layer
                layers += [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3], # access the tuple to get the kernel size, output size, stride and padding -> note we are using **kwargs here (kernel_size, stride, padding)
                    )
                ]
                in_channels = x[1]

            elif type(x) == str: # If its a string, its a Maxpool layer
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

            elif type(x) == list: # If its a list, its multiple conv layers repeated
                conv1 = x[0] # first conv layer
                conv2 = x[1] # second conv layer
                num_repeats = x[2] # how many times these conv layers are repeated after each other

                for _ in range(num_repeats):
                    layers += [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]
                    layers += [
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]
                    in_channels = conv2[1]

        return nn.Sequential(*layers) # Convert the list of layers into a nn.Sequential object (basically a list of nn modules)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(), # for sanity, we can make sure it is flattened
            nn.Linear(1024 * S * S, 496), # Here we will downsize to reduce memory usage and speed up training
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(496, S * S * (C + B * 5)), # rmb output dimensionality depends on the NxN grid size we choose, how many classes we want to detect and how many bounding boxes per cell we want
        )