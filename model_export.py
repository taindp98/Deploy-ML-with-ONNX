# Global imports
import argparse
import os
import numpy as np
import torch 
import torch.nn as nn
import collections

# Package imports
from engine import albu_transform as albu_fork
import albumentations as A
import os.path as osp
import cv2
import json
from utils import *
from tqdm import tqdm
import onnxruntime
import onnx
from torchvision import models
class ModelExporter(nn.Module):
    def __init__(self, file_configs) -> None:
        super().__init__()

        # Load config
        default_config, trial_config = file_configs
        default_config, tuple_key_list = load_config(default_config)
        trial_config, _ = load_config(trial_config, tuple_key_list=tuple_key_list)
        self.config = {**default_config, **trial_config}
        self.device = torch.device(self.config['device'])

        model, model_without_ddp = get_model(config = self.config, num_pid=None, device=self.device)        
        self.model = model
        self.model.eval()

        
        ## detection score
        self.det_thres = 0.5

        self.tfs_fnc_query = A.Compose([
                    albu_fork.WindowResize(min_size = 900, max_size = 1500),
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ],
                    bbox_params=A.BboxParams(format='coco'))

        self.tfs_fnc_gallery = A.Compose([
                    albu_fork.WindowResize(min_size = 900, max_size = 1500),
                    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
    
    def to_onnx(self, h = 900, w = 1500):
        self.model.forward = self.model.fast_inference
        self.model.eval()
        self.dummy_img = torch.randn(1,3,h,w).to(self.device)
        self.dummy_target = torch.FloatTensor([[0., 0., 0., 1.]]).to(self.device)
        args = self.dummy_img, self.dummy_target
        self.dummy_output = self.model(*args)[0]
        print(self.dummy_output)
        input_names = ['images', 'targets']
        output_names = ['output']
        torch.onnx.export(self.model,
                args = (self.dummy_img, self.dummy_target), # model input (or a tuple for multiple inputs)  , self.dummy_target
                f = "./checkpoints/seqnext.onnx",
                verbose=False,
                input_names=input_names,
                output_names=output_names,
                export_params=True,
                opset_version=13,
                do_constant_folding=True,
                dynamic_axes={'images' : {0 : 'batch_size', 1: 'channel', 2:'height', 3:'width'}, 
                            'targets': {0: 'batch_size', 1: 'coor'},
                            'output' : {0 : 'batch_size'}
                            }
                 )
        print('Model has been converted to ONNX') 

    def to_numpy(self, tensor, is_img = False):
        if is_img:
            return [tensor[0].detach().cpu().numpy() if tensor[0].requires_grad else tensor[0].cpu().numpy()]
        else:
            return tensor.cpu().numpy()
    
    def run_onnx(self):

        onnx_model = onnx.load('./checkpoints/seqnext.onnx')
        # print(onnx.checker.check_model(onnx_model))
        print('check input name ',onnx_model.graph.input)
        print('check output name ',onnx_model.graph.output)
        # print(onnx.helper.printable_graph(onnx_model.graph))
        try:
            onnx.checker.check_model(onnx_model)
        except onnx.checker.ValidationError as e:
            print("The model is invalid: %s" % e)
        else:
            print("The model is valid!")
        ort_sess = onnxruntime.InferenceSession('./checkpoints/seqnext.onnx')
        # print(ort_sess.get_inputs())
        outputs = ort_sess.run([], {ort_sess.get_inputs()[0].name: self.to_numpy(self.dummy_img, True),
                                    ort_sess.get_inputs()[1].name: self.to_numpy(self.dummy_target)
                                    })
        print(outputs[0])
        test_offset = np.testing.assert_allclose(self.dummy_output.detach().cpu().numpy(), outputs[0], rtol=1e-03, atol=1e-05)
        print(test_offset)

def main():
    parser = argparse.ArgumentParser(description="Inference module of Person Search project.")
    parser.add_argument("--file_configs", dest = 'file_configs', help="Path to configuration file.", default='./configs/test.yaml')
    args = parser.parse_args()
    exporter = ModelExporter(file_configs = ('./configs/default.yaml', args.file_configs))
    
    ## PRW/MMPTrack
    DATA_DIR = './demo'
    exporter.to_onnx()
    exporter.run_onnx()
if __name__ == '__main__':
    main()