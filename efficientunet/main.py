import torch
from SDFY_project.efficientunet.efficientynet import *

# This is a demo to show you how to use the library.
if __name__ == '__main__':
    t = torch.rand(2, 3, 224, 224).cuda()

    # EfficientNet test
    # model = EfficientNet.from_name('efficientnet-b3', n_classes=5, pretrained=True).cuda()
    # print(model(t).size())

    # EfficientNet with custom head test
    # model_ch = EfficientNet.custom_head('efficientnet-b5', n_classes=5, pretrained=True).cuda()
    # print(model_ch(t).size())

    # EfficientUnet test
    b0unet = get_efficientynet_b3(out_channels=1, n_classes=5,concat_input=True, pretrained=True).cuda()
    print(b0unet(t))
