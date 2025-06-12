# -*- coding: utf-8 -*-
from thop import profile

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total: %.2fM' % (total_num / 1e6), 'Trainable_num: %.2fM' % (trainable_num / 1e6)}


def get_params_and_flops(net, inputs):
    flops, params = profile(net, inputs)
    return {'FLOPs = ' + str((flops/1000**3)/32) + 'G', 'Params = ' + str(params/1000**2) + 'M'}