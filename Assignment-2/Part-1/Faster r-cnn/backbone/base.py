from typing import Tuple, Type

from torch import nn


class Base(object):

    OPTIONS = ['resnet101']

    @staticmethod
    def from_name(name: str) -> Type['Base']:
        if name == 'resnet101':
            from backbone.resnet101 import ResNet101
            return ResNet101
        else:
            raise ValueError

    def __init__(self, pretrained: bool):
        super().__init__()
        self._pretrained = pretrained

    def features(self) -> Tuple[nn.Module, nn.Module, int, int]:
        raise NotImplementedError
