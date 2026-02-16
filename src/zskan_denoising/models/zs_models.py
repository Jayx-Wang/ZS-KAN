import torch.nn as nn

from zskan_denoising.layers.kan_conv_v1.KANConv import KAN_Convolutional_Layer


class ZS_N2N(nn.Module):
    def __init__(self, n_chan: int = 3, chan_embed: int = 48):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv4 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        return self.conv4(x)


class ZS_MKAN(nn.Module):
    def __init__(self, n_chan: int = 3, chan_embed: int = 12, grid_size: int = 3, device: str = "cuda"):
        super().__init__()
        self.kan_conv1 = KAN_Convolutional_Layer(
            n_convs=2,
            grid_size=grid_size,
            kernel_size=(3, 3),
            padding=(1, 1),
            device=device,
        )
        self.kan_conv2 = KAN_Convolutional_Layer(
            n_convs=2,
            grid_size=grid_size,
            kernel_size=(3, 3),
            padding=(1, 1),
            device=device,
        )
        self.conv = nn.Conv2d(chan_embed, n_chan, 1)
        self.kan_conv6 = KAN_Convolutional_Layer(n_convs=1, kernel_size=(1, 1), device=device)

    def forward(self, x):
        x = self.kan_conv1(x)
        x = self.kan_conv2(x)
        x = self.conv(x)
        return self.kan_conv6(x)


class ZS_KAN(nn.Module):
    def __init__(self, n_chan: int = 3, chan_embed: int = 25, grid_size: int = 3, device: str = "cuda"):
        super().__init__()
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv = nn.Conv2d(chan_embed, n_chan, 3, padding=1)
        self.kan_conv = KAN_Convolutional_Layer(
            n_convs=1,
            grid_size=grid_size,
            kernel_size=(1, 1),
            device=device,
        )

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv(x))
        return self.kan_conv(x)


model_dict = {
    "zs_n2n": ZS_N2N,
    "zs_kan": ZS_KAN,
    "zs_mkan": ZS_MKAN,
}


def build_model(model_name: str, n_chan: int, device: str):
    key = model_name.lower()
    if key not in model_dict:
        raise ValueError(f"Unsupported model '{model_name}'. Choices: {list(model_dict)}")
    if key == "zs_n2n":
        return model_dict[key](n_chan)
    return model_dict[key](n_chan, device=device)
