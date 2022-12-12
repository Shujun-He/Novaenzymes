import torch.nn as nn

class ThermoNet2(nn.Module):
    def __init__(self, params):
        super().__init__()

        CONV_LAYER_SIZES = [14, 16, 24, 32, 48, 78, 128]
        FLATTEN_SIZES = [0, 5488, 5184, 4000, 3072, 2106, 1024]

        dropout_rate = params['dropout_rate']
        dropout_rate_dt = params['dropout_rate_dt']
        dense_layer_size = int(params['dense_layer_size'])
        layer_num = int(params['conv_layer_num'])
        silu = params['SiLU']

        self.params = params
        if silu:
            activation = nn.SiLU()
        else:
            activation = nn.ReLU()

        model = [
            nn.Sequential(
                *[nn.Sequential(
                    nn.Conv3d(in_channels=CONV_LAYER_SIZES[l], out_channels=CONV_LAYER_SIZES[l + 1], kernel_size=(3, 3, 3)),
                    activation
                ) for l in range(layer_num)]
            ),
            nn.MaxPool3d(kernel_size=(2,2,2)),
            nn.Flatten(),
        ]
        flatten_size = FLATTEN_SIZES[layer_num]
        if self.params['LayerNorm']:
            model.append(nn.LayerNorm(flatten_size))
        self.model = nn.Sequential(*model)

        self.ddG = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=flatten_size, out_features=dense_layer_size),
            activation,
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=dense_layer_size, out_features=1)
        )
        self.dT = nn.Sequential(
            nn.Dropout(p=dropout_rate_dt),
            nn.Linear(in_features=flatten_size, out_features=dense_layer_size),
            activation,
            nn.Dropout(p=dropout_rate_dt),
            nn.Linear(in_features=dense_layer_size, out_features=1)
        )


    def forward(self, x):
        if self.params['diff_features']:
            x[:, 7:, ...] -= x[:, :7, ...]
        x = self.model(x)
        ddg = self.ddG(x)
        dt = self.dT(x)
        return ddg.squeeze(), dt.squeeze()
