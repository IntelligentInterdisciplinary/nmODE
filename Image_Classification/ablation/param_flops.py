import torch
from torchsummary import summary
from ptflops import get_model_complexity_info
from torchdiffeq import odeint

class Config:
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset="fashionmnist"#mnist,cifar10,fashionmnist
    epoch = 300
    batchsize = 200
    num_workers = 8
    torch.manual_seed(42)

    alpha = 1e-3
    num_classes = 10
    input_dim = 28*28
    lam = 1.0
    rtol = 1e-3
    atol = 1e-3

    hidden_dim = 4096
    delta = 1.0

class neuralODE(torch.nn.Module):
    def __init__(self):
        super(neuralODE, self).__init__()
    def fresh(self, gamma):
        self.gamma = gamma
    def forward(self, t, y):
        dydt = -config.lam * y + torch.pow(torch.sin(y+self.gamma),2)
        return dydt

class nmODEblock(torch.nn.Module):
    def __init__(self):
        super(nmODEblock, self).__init__()
        self.LP1 = torch.nn.Linear(config.input_dim,config.hidden_dim)
        self.LP2 = torch.nn.Linear(config.input_dim,config.hidden_dim)
        self.LP3 = torch.nn.Linear(config.hidden_dim,10)
        self.neuralODE = neuralODE()
        self.t = torch.tensor([0,config.delta], device=config.device)
    def forward(self,x,y):
        x = x.view(-1,config.input_dim)
        gamma = self.LP1(x)
        self.neuralODE.fresh(gamma)
        y = odeint(self.neuralODE, y, self.t, method='dopri5', rtol=config.rtol, atol=config.atol)[-1]
        z = self.LP3(self.LP2(x) * y)
        return z,y

config = Config()
model = nmODEblock().to(config.device)

# 创建一个包装器类
class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        print(x.shape)
        x1, x2 = torch.split(x, [config.input_dim, config.hidden_dim], dim=2)  # 将输入分解为两个部分
        return self.model(x1, x2)

wrapped_model = Wrapper(model).to(config.device)

# 计算参数量
params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
print(f'Total trainable parameters: {params}')

# 计算FLOPs
with torch.cuda.device(0):
  flops, params = get_model_complexity_info(wrapped_model, (config.batchsize, config.input_dim + config.hidden_dim), as_strings=True, print_per_layer_stat=False)
  print(f'Total FLOPs: {flops}')


# from torchsummary import summary
# from thop import profile
# import torch
# from torch.nn import functional as F
# from torchdiffeq import odeint
# from torch import nn
# from collections import OrderedDict
# import numpy as np

# class Config:
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     input_dim = 28*28
#     hidden_dim = 4096
#     delta = 0.05
#     lam = 1.0
#     rtol = 1e-3
#     atol = 1e-3
#     batch_size = 200

# config = Config()

# class neuralODE(torch.nn.Module):
#     def __init__(self):
#         super(neuralODE, self).__init__()
#     def fresh(self, gamma):
#         self.gamma = gamma
#     def forward(self, t, y):
#         dydt = -config.lam * y + torch.pow(torch.sin(y+self.gamma),2)
#         return dydt

# class nmODEblock(torch.nn.Module):
#     def __init__(self):
#         super(nmODEblock, self).__init__()
#         self.LP1 = torch.nn.Linear(config.input_dim,config.hidden_dim)
#         self.LP2 = torch.nn.Linear(config.input_dim,config.hidden_dim)
#         self.LP3 = torch.nn.Linear(config.hidden_dim,10)
#         self.neuralODE = neuralODE()
#         self.t = torch.tensor([0,config.delta], device=config.device)
#     def forward(self,x,y):
#         x = x.view(-1,config.input_dim)
#         gamma = self.LP1(x)
#         self.neuralODE.fresh(gamma)
#         y = odeint(self.neuralODE, y, self.t, method='dopri5', rtol=config.rtol, atol=config.atol)[-1]
#         z = self.LP3(self.LP2(x) * y)
#         return z,y

# def multiple_input_summary(model, input_sizes, device):
#     def register_hook(module):
#         def hook(module, input, output):
#             class_name = str(module.__class__).split(".")[-1].split("'")[0]
#             module_idx = len(summary)

#             m_key = f"{class_name}-{module_idx + 1}"
#             summary[m_key] = {}
#             summary[m_key]["input_shape"] = [size for size in input[0].size()]
#             summary[m_key]["output_shape"] = [size for size in output.size()]
#             summary[m_key]["nb_params"] = sum(p.numel() for p in module.parameters())

#         if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
#             hooks.append(module.register_forward_hook(hook))

#     device = device.lower()
#     assert device in ["cuda", "cpu"], "Input device is not valid, please specify 'cuda' or 'cpu'"

#     if device == "cuda" and torch.cuda.is_available():
#         dtype = torch.cuda.FloatTensor
#     else:
#         dtype = torch.FloatTensor

#     # check if there are multiple inputs to the network
#     if isinstance(input_sizes[0], (list, tuple)):
#         x = [torch.rand(2, *in_size).type(dtype) for in_size in input_sizes]
#     else:
#         x = torch.rand(2, *input_sizes).type(dtype)
#         x = x.to(device)
#         x = (x,)

#     # create properties
#     summary = OrderedDict()
#     hooks = []

#     # register hook
#     model.apply(register_hook)

#     # make a forward pass
#     model(*x)

#     # remove these hooks
#     for h in hooks:
#         h.remove()

#     print("----------------------------------------------------------------")
#     line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
#     print(line_new)
#     print("================================================================")
#     total_params = 0
#     total_output = 0
#     trainable_params = 0
#     for layer in summary:
#         # input_shape, output_shape, trainable, nb_params
#         line_new = "{:>20}  {:>25} {:>15}".format(
#             layer,
#             str(summary[layer]["output_shape"]),
#             "{0:,}".format(summary[layer]["nb_params"]),
#         )
#         total_params += summary[layer]["nb_params"]
#         total_output += np.prod(summary[layer]["output_shape"])
#         if "trainable" in summary[layer]:
#             if summary[layer]["trainable"] == True:
#                 trainable_params += summary[layer]["nb_params"]
#         print(line_new)

#     # assume 4 bytes/number (float on cuda).
#     total_input_size = abs(np.prod(input_sizes) * config.batch_size * 4. / (1024 ** 2.))
#     total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
#     total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
#     total_size = total_params_size + total_output_size + total_input_size

#     print("================================================================")
#     print("Total params: {0:,}".format(total_params))
#     print("Trainable params: {0:,}".format(trainable_params))
#     print("Non-trainable params: {0:,}".format(total_params - trainable_params))
#     print("----------------------------------------------------------------")
#     print("Input size (MB): %0.2f" % total_input_size)
#     print("Forward/backward pass size (MB): %0.2f" % total_output_size)
#     print("Params size (MB): %0.2f" % total_params_size)
#     print("Estimated Total Size (MB): %0.2f" % total_size)
#     print("----------------------------------------------------------------")
#     # return summary

# model = nmODEblock().to(config.device)
# input_size = (1, config.input_dim)
# hidden_size = (1, config.hidden_dim)
# y = torch.zeros(*hidden_size).to(config.device)

# device = 'cuda' if 'cuda' in str(config.device) else 'cpu'
# # Use the custom summary function for multiple inputs
# multiple_input_summary(model, [input_size, hidden_size], device=device)

# input = torch.randn(*input_size).to(config.device)
# macs, params = profile(model, inputs=(input, y))
# print(f"MACs: {macs}")
# print(f"Params: {params}")

# # from torchsummary import summary
# # from thop import profile
# # import torch
# # from torch.nn import functional as F
# # from torchdiffeq import odeint
# # from torch import nn

# # class Config:
# #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# #     input_dim = 28*28
# #     hidden_dim = 4096
# #     delta = 0.05
# #     lam = 1.0
# #     rtol = 1e-3
# #     atol = 1e-3

# # config = Config()

# # class neuralODE(torch.nn.Module):
# #     def __init__(self):
# #         super(neuralODE, self).__init__()
# #     def fresh(self, gamma):
# #         self.gamma = gamma
# #     def forward(self, t, y):
# #         dydt = -config.lam * y + torch.pow(torch.sin(y+self.gamma),2)
# #         return dydt

# # class nmODEblock(torch.nn.Module):
# #     def __init__(self):
# #         super(nmODEblock, self).__init__()
# #         self.LP1 = torch.nn.Linear(config.input_dim,config.hidden_dim)
# #         self.LP2 = torch.nn.Linear(config.input_dim,config.hidden_dim)
# #         self.LP3 = torch.nn.Linear(config.hidden_dim,10)
# #         self.neuralODE = neuralODE()
# #         self.t = torch.tensor([0,config.delta], device=config.device)
# #     def forward(self,x,y):
# #         x = x.view(-1,config.input_dim)
# #         gamma = self.LP1(x)
# #         self.neuralODE.fresh(gamma)
# #         y = odeint(self.neuralODE, y, self.t, method='dopri5', rtol=config.rtol, atol=config.atol)[-1]
# #         z = self.LP3(self.LP2(x) * y)
# #         return z,y

# # model = nmODEblock().to(config.device)
# # input_size = (1, config.input_dim)
# # hidden_size = (1, config.hidden_dim)
# # y = torch.zeros(*hidden_size).to(config.device)

# # device = 'cuda' if 'cuda' in str(config.device) else 'cpu'
# # summary(model, [input_size, hidden_size], device=device)

# # input = torch.randn(*input_size).to(config.device)
# # macs, params = profile(model, inputs=(input, y))
# # print(f"MACs: {macs}")
# # print(f"Params: {params}")