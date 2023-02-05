import yaml
import torch 
from models.seqnext import SeqNeXt
from datetime import datetime,date

def load_config(path, tuple_key_list=None):
    # Load config dict from YAML
    with open(path, 'r') as fp:
        try:
            config = yaml.safe_load(fp)
        except yaml.YAMLError as exc:
            print(exc)
            raise

    # Params that have tuple type
    if tuple_key_list is None:
        tuple_key_list = config['tuple_key_list']
        del config['tuple_key_list']
    elif 'tuple_key_list' in config:
        del config['tuple_key_list']
    
    # Convert lists to tune.grid_search
    proc_config = {}
    for key, val in config.items():
        if key in tuple_key_list:
            if type(val) == list:
                proc_config[key] = [eval(_val) for _val in val]
            else:
                proc_config[key] = eval(val)
        else:
            proc_config[key] = val

    # Return processed config dict
    return proc_config, tuple_key_list

def get_model(config, num_pid = None, device='cpu'):

    def _del_key(state_dict, key):
        if key in state_dict:
            del state_dict[key]
    
    # Build SeqNeXt model
    model = SeqNeXt(config, oim_lut_size = num_pid, device=device)
    # Put model on GPU
    print('Cuda available:', torch.cuda.is_available())
    model.to(device)

    model_without_ddp = model

    # Load checkpoint if it is available
    if config['ckpt_path']:
        print('==> Restoring checkpoint from path:', config['ckpt_path'])
        checkpoint = torch.load(config['ckpt_path'], map_location='cpu')
        state_dict = checkpoint['model']
        # Delete keys with potentially conflicting params
        _del_key(state_dict, 'roi_heads.reid_loss.lut')
        _del_key(state_dict, 'roi_heads.reid_loss.cq')
        _del_key(state_dict, 'gfn.reid_loss.lut')
        _del_key(state_dict, 'gfn.reid_loss.cq')
        _del_key(state_dict, 'roi_heads.gfn.reid_loss.lut')
        _del_key(state_dict, 'roi_heads.gfn.reid_loss.cq')
        # Load state dict into the model
        model_without_ddp.load_state_dict(state_dict, strict=False)

    return model, model_without_ddp


def get_runtime():
    d = date.today().strftime("%m_%d_%Y") 
    h = datetime.now().strftime("%H_%M_%S").split('_')
    h_offset = int(datetime.now().strftime("%H_%M_%S").split('_')[0]) + 7 ## vietnam GMT +7
    h[0] = str(h_offset)
    h = '_'.join(h)
    run_time =  d +'_' + h
    return run_time