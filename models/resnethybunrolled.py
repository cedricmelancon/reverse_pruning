# Hybrid ResNet - BasicBlock - binary block of 2 conv layers, BasicBlock2 - full-precision block of 2 conv layers.
import tensorflow as tf

import pdb
__all__ = ['resnethybunrolled']






def resnethybunrolled(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    if dataset == 'imagenet':
        num_classes = num_classes or 1000
        depth = depth or 50
        if depth == 18:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[2, 2, 2, 2])
        if depth == 34:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=BasicBlock, layers=[3, 4, 6, 3])
        if depth == 50:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 6, 3])
        if depth == 101:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 4, 23, 3])
        if depth == 152:
            return ResNet_imagenet(num_classes=num_classes,
                                   block=Bottleneck, layers=[3, 8, 36, 3])

    elif dataset == 'cifar10':
        num_classes = num_classes or 10
        depth = depth or 18
        return ResNet_cifar10(num_classes=num_classes,
                              block=BasicBlock, depth=depth)
    elif dataset == 'cifar100':
        num_classes = num_classes or 100
        depth = depth or 18
        return ResNet_cifar100(num_classes=num_classes,
                               block=BasicBlock, depth=depth)
