from sklearn.base import BaseEstimator as SklearnBaseEstimator
from torch import nn
import torch

import numpy as np

import functools
import operator


class BaseEstimator(SklearnBaseEstimator):
    # http://msmbuilder.org/development/apipatterns.html

    def summarize(self):
        return 'NotImplemented'


def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def load_model_from_path(self, path, device):
        # load model
        map_location = f"{device.type}:{device.index}"
        if device.type == 'cpu':
            map_location = device.type

        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint['model_state_dict'])

        return self.to(device)

    @staticmethod
    def num_features_before_fcnn(model, input_dim):
        return functools.reduce(operator.mul,
                                list(model(torch.rand(1, *input_dim)).shape))

    def summary(model, input_size, batch_size=-1, device="cuda"):
        '''Wrapper for https://github.com/sksq96/pytorch-summary'''

        def register_hook(module):

            def hook(module, input, output):

                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)

                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = dict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
                        [-1] + list(o.size())[1:] for o in output
                    ]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(
                        torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(
                        torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params

            if (not isinstance(module, nn.Sequential) and
                    not isinstance(module, nn.ModuleList) and
                    not (module == model)):
                hooks.append(module.register_forward_hook(hook))

        device = device.lower()
        assert device in [
            "cuda",
            "cpu",
        ], "Input device is not valid, please specify 'cuda' or 'cpu'"

        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor

        if isinstance(input_size, dict):
            x = dict()
            for in_key, in_size in input_size.items():
                x[in_key] = torch.rand(2, *in_size).type(dtype)
            x = [x]

        else:
            # multiple inputs to the network
            if isinstance(input_size, tuple):
                input_size = [input_size]
            # batch_size of 2 for batchnorm
            x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]

        # create properties
        summary = dict()
        hooks = []

        # register hook
        model.apply(register_hook)

        # make a forward pass
        # print(x.shape)

        model(*x)

        # remove these hooks
        for h in hooks:
            h.remove()

        print(
            "----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>25} {:>15}".format(
            "Layer (type)", "Input Shape", "Output Shape", "Param #")
        print(line_new)
        print(
            "================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            # input_shape, output_shape, trainable, nb_params
            line_new = "{:>20}  {:>25} {:>25} {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print(line_new)

        # summary_df = {'layer': layer,
        #               'in_shape': summary[layer]["input_shape"],
        #               'out_shape': summary[layer]["output_shape"],
        #               'params': summary[layer]["nb_params"]
        #               for layer in summary}

        # assume 4 bytes/number (float on cuda).
        if isinstance(input_size, dict):
            total_input_size = 0
            for in_key, in_size in input_size.items():
                total_input_size += abs(
                    np.prod(in_size) * batch_size * 4. / (1024**2.))

        else:
            total_input_size = abs(
                np.prod(input_size) * batch_size * 4. / (1024**2.))
        total_output_size = abs(2. * total_output * 4. /
                                (1024**2.))  # x2 for gradients
        total_params_size = abs(total_params.numpy() * 4. / (1024**2.))
        total_size = total_params_size + total_output_size + total_input_size

        print(
            "================================================================")
        print("Total params: {0:,}".format(total_params))
        print("Trainable params: {0:,}".format(trainable_params))
        print("Non-trainable params: {0:,}".format(total_params -
                                                   trainable_params))
        print(
            "----------------------------------------------------------------")
        print("Input size (MB): %0.2f" % total_input_size)
        print("Forward/backward pass size (MB): %0.2f" % total_output_size)
        print("Params size (MB): %0.2f" % total_params_size)
        print("Estimated Total Size (MB): %0.2f" % total_size)
        print(
            "----------------------------------------------------------------")
        return summary