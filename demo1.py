#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project LED
@File demo1.py
@Date 2022/7/6 17:28
@Author MIaocunke
@Email Miaocunke@moweai.com
@Desc 
"""
from torchvision import models
model = models.resnet101(pretrained=True)
print(model)