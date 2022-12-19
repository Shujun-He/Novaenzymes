import torch.nn as nn
import torch

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

from e3nn.nn.models.gate_points_2101 import Network
from e3nn import o3



class e3nnNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        model_kwargs = {
            "irreps_in": "25x 0e",
            "irreps_hidden": [(mul, (l, p)) for l, mul in enumerate([10,3,2,1]) for p in [-1, 1]],
            "irreps_out": "256x0e",
            "irreps_node_attr": None,
            "irreps_edge_attr": o3.Irreps.spherical_harmonics(3),
            "layers": 1,
            "max_radius": 20,
            "number_of_basis": 10,
            "radial_layers": 1,
            "radial_neurons": 128,
            "num_neighbors": 32,
            "num_nodes": 100,
            "reduce_output": False,
        }

        # model_kwargs = {
        #     "irreps_in": "25x 0e",
        #     "irreps_hidden": [(mul, (l, p)) for l, mul in enumerate([20,6,4,2]) for p in [-1, 1]],
        #     "irreps_out": "256x0e",
        #     "irreps_node_attr": None,
        #     "irreps_edge_attr": o3.Irreps.spherical_harmonics(3),
        #     "layers": 1,
        #     "max_radius": 20,
        #     "number_of_basis": 10,
        #     "radial_layers": 1,
        #     "radial_neurons": 128,
        #     "num_neighbors": 32,
        #     "num_nodes": 100,
        #     "reduce_output": False,
        # }

        self.model = Network(**model_kwargs)
        self.linear = nn.Linear(256,2)

    # def forward(self,input):
    #     wt_input={"pos":input['wt_pos'],
    #               "x":input['wt_x'],
    #               'batch':input['wt_batch']}
    #     wt=self.model(wt_input)
    #
    #     mt_input={"pos":input['mt_pos'],
    #               "x":input['mt_x'],
    #               'batch':input['mt_batch']}
    #     mt=self.model(mt_input)
    #
    #     max_index=int(wt_input['batch'].max().item())
    #
    #     output=[]
    #
    #     for batch_index in range(max_index+1):
    #
    #         wt_sample=wt[wt_input['batch']==batch_index].mean(0)
    #         mt_sample=mt[mt_input['batch']==batch_index].mean(0)
    #         # dt_output.append(mt_sample[0]-wt_sample[0])
    #         # ddg_output.append(mt_sample[1]-wt_sample[1])
    #         output.append(mt_sample-wt_sample)
    #
    #     output=torch.stack(output,0)
    #
    #     #print(output.shape)
    #
    #     dt_output=output[:,0]
    #     ddg_output=output[:,1]
    #
    #     # print(dt_output.shape)
    #     # print(ddg_output.shape)
    #     # exit()
    #
    #     return dt_output.float(), ddg_output.float()

    def forward(self,input):
        input={"pos":torch.cat([input['wt_pos'],input['mt_pos']],0),
                "x":torch.cat([input['wt_x'],input['mt_x']],0),
                'batch':torch.cat([input['wt_batch'],input['mt_batch']],0).long()}

        # print(input['pos'].shape)
        # print(input['x'].shape)
        # print(input['batch'].shape)
        # exit()
        y=self.model(input)
        #y=self.linear(y)
        max_index=int(input['batch'].max().item())

        output=[]
        for batch_index in range(max_index+1):
            sample=y[(input['batch']==batch_index)*(input['x'][:,-1]==1)].mean(0)
            sample=sample-y[(input['batch']==batch_index)*(input['x'][:,-1]==0)].mean(0)
            #sample=y[input['batch']==batch_index].mean(0)
            output.append(sample)

        output=torch.stack(output,0)
        output=self.linear(output)
        #print(output.shape)

        dt_output=output[:,0]
        ddg_output=output[:,1]

        # print(dt_output.shape)
        # print(ddg_output.shape)
        # exit()

        return dt_output.float(), ddg_output.float()
