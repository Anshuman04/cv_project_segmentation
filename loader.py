"""
This file is responsible to load the model on the fly for the model configs
given in model_configs/<model structure>.json
"""

import os
import json
import logging
import torch
import collections

import torch.nn as nn
import torch.nn.functional as F

CONFIG_FOLDER = "model_configs"



class ModelLoader(nn.Module):
    def __init__(self, modelConfig):
        super(ModelLoader, self).__init__()

        self.fileName = modelConfig
        self._configHomePath = None
        self._configFilePath = None
        self.data = self.loadData()
        self.lookUpDict = self._lookUpDict()
        import pdb
        pdb.set_trace()
        self.initLayers()
        # self.forward = self.forwardPropogation

    @property
    def configHomePath(self):
        if not self._configHomePath:
            self._configHomePath = os.path.join(os.getcwd(), CONFIG_FOLDER)
        return self._configHomePath

    @property
    def configFilePath(self):
        if not self._configFilePath:
            self._configFilePath = os.path.join(self.configHomePath, self.fileName)
        return self._configFilePath
    
    def _lookUpDict(self):
        lookUpDict = {
            "fc": nn.Linear,
            "conv": nn.Conv2d,
            "drop": nn.Dropout,
            "sigmoid": torch.sigmoid,
            "relu": torch.relu,
            "maxPool": F.max_pool2d,
            "flatten": self.flatten
        }
        return lookUpDict
    
    def loadData(self):
        data = {}
        with open(self.configFilePath, "r") as fp:
            data = json.load(fp)
        return data

    def flatten(self, X):
        return X.view(-1, self.num_flat_features(X))
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def initLayers(self):
        layerNum = 1
        for layerData in self.data.get("layers"):
            layerVar = "{}_{}".format(layerData.get("type", "undefined"), layerNum)
            params = layerData.get("params")
            if isinstance(params, list):
                layerVal = self.lookUpDict.get(layerData.get("type"))(*params)
            elif isinstance(params, dict):
                layerVal = self.lookUpDict.get(layerData.get("type"))(**params)
            elif params is None:
                layerNum += 1
                continue
            else:
                raise Exception("Invalid config format in layerData")
            setattr(self, layerVar, layerVal)
            self.data["layers"][layerNum-1]["layerObj"] = getattr(self, layerVar)
            layerNum += 1

    def forward(self, X):
        layerNum = 1
        for layerData in self.data.get("layers"):
            if layerData["type"] == "flatten":
                X = self.lookUpDict.get(layerData.get("type"))(X)
                continue
            X = layerData["layerObj"](X)
            if layerData.get("activation"):
                X = self.lookUpDict[layerData["activation"]](X)
            if layerData.get("postProc"):
                postProcMethod = self.lookUpDict[layerData["postProc"]["type"]]
                postProcParams = layerData.get("postProc")["params"]
                if isinstance(postProcParams, list):
                    X = postProcMethod(X, *postProcParams)
                elif isinstance(postProcParams, dict):
                    X = postProcMethod(X, **postProcParams)
                else:
                    raise Exception("Invalid config format in post processing")
        return X








