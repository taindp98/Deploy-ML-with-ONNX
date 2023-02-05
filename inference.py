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

class PersonSearchInference(nn.Module):
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
    
    def forward(self, img: str, box = None, is_coco = False):
        """
        forward method receives single image
        """
        img = cv2.imread(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if box:
            ## query
            box = np.array(box)
            if not is_coco:
                box[2:] -= box[:2]
            box = np.append(box, -1)  ## dummy label because coco format
            
            out_tfs = self.tfs_fnc_query(image = img, bboxes = [box])

            ## fixbug: SeqNet receives the box as xmin, ymin, xmax, ymax. NOT COCO
            box_tfs = np.array(out_tfs['bboxes'][0][:4])
            box_tfs[2:] += box_tfs[:2]
            box_tfs = np.expand_dims(box_tfs, 0).astype(np.float32)
            box_tfs = torch.tensor(box_tfs)
            box_tfs = [{"boxes": box_tfs.to(self.device)}]

            img_tfs = out_tfs['image']
            img_tfs = np.expand_dims(img_tfs, 0)
            img_tfs = torch.permute(torch.tensor(img_tfs), (0,3,1,2))
            img_tfs = img_tfs.to(self.device)
            out = self.model(images = img_tfs, targets = box_tfs, inference_mode = 'gt')[1][0]  ## embeddings have ids 1
            return out
        
        else:
            ## gallery
            out_tfs = self.tfs_fnc_gallery(image = img)
            img_tfs = out_tfs['image']
            img_tfs = np.expand_dims(img_tfs, 0)
            img_tfs = torch.permute(torch.tensor(img_tfs), (0,3,1,2))
            img_tfs = img_tfs.to(self.device)
            out = self.model(images = img_tfs, inference_mode = 'det')[0][0]  ## detections have ids 0

            det_scores = out['scores'].cpu().data.numpy()
            high_conf_ids = np.where(det_scores > self.det_thres)[0].tolist()
            ## predicted bounding boxes of query image
            det = out['boxes'].cpu().data.numpy().astype(int)[high_conf_ids]
            feat = out['embeddings'][high_conf_ids].cpu().data.numpy()
            ## need to crop the RoI
            
            return (det, feat)    

def main():
    parser = argparse.ArgumentParser(description="Inference module of Person Search project.")
    parser.add_argument("--file_configs", dest = 'file_configs', help="Path to configuration file.", default='./configs/test.yaml')
    args = parser.parse_args()
    processor = PersonSearchInference(file_configs = ('./configs/default.yaml', args.file_configs))
    
    ## PRW/MMPTrack
    DATA_DIR = './demo'
    
    query_img  = osp.join(DATA_DIR, 'rgb_00000_1.jpg')
    box = [337, 33, 385, 130]  ## pid = 1
    emb = processor(query_img, box, is_coco = False)
    print(emb)
    det, feat = processor(query_img)
    print(det)

    ## MMPTrack
    # detection_lookup = {}
    # DATA_DIR = processor.config['dataset_dir']
    # data_query = json.load(open(osp.join(DATA_DIR, 'resources', 'df_test_q100_g50_pid_all.json')))
    # _idx = str(0)
    # gallery_id = data_query['gallery'][_idx]
    # gallery_img = [osp.join(DATA_DIR, 'mmptrack_prw', item + '.jpg') for item in gallery_id]
    # for g_id, g_img in tqdm(zip(gallery_id, gallery_img), total = len(gallery_img)):
    #     det, feat = processor(g_img)
    #     detection_lookup[g_id] = {'boxes': det, 'embeddings': feat}

    # torch.save(detection_lookup, osp.join(processor.config['log_dir'], f'gallery_vectors_{_idx}.pth'))
    # print('Inference is successful')

if __name__ == '__main__':
    main()