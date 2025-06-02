import torch
#from models import vision_transformer as vit
#from models import vision_transformer_multiBlocks_20221030 as vit
#from methods import vision_transformer_multiBlocks_20221030 as vit
from methods import ViT as vit
#import vision_transformer_multiBlocks_20221030 as vit
#from models.pmf_protonet import ProtoNet
#from methods.pmf_protonet import ProtoNet
from methods.protonet import ProtoNet
#from pmf_protonet import ProtoNet
#from models.cvpr2023_gnnnet_20221102 import GnnNet
#from methods.cvpr2023_gnnnet_20221102 import GnnNet
#from cvpr2023_gnnnet_20221102 import GnnNet

def load_ViTsmall(no_pretrain=False):
  model = vit.__dict__['vit_small'](patch_size=16, num_classes=0)
  if(not no_pretrain):
    # check if the local path exists
    local_path = "./checkpoints/pretrained_models/dino_deitsmall16_pretrain.pth"
    import os
    if os.path.exists(local_path):
      # if the local file exists, load the pretrained weight from local
      state_dict = torch.load(local_path, map_location="cpu")
      print(f'load the pretrained weight from local path: {local_path}')
    else:
      # if the local file does not exist, download the pretrained weight from network
      url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
      try:
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        print(f'download the pretrained weight from network')
      except Exception as e:
        print(f'download the pretrained weight from network failed: {e}')
        print(f'please download the model file to: {local_path}')
        raise e
    
    model.load_state_dict(state_dict, strict=True)
  #print('model defined.')
  return model

def load_ViTbase(no_pretrain=False):
  model = vit.__dict__['vit_base'](patch_size=16, num_classes=0)
  if(not no_pretrain):
    # check if the local path exists
    local_path = "./checkpoints/pretrained_models/dino_vitbase16_pretrain.pth"
    import os
    if os.path.exists(local_path):
      # if the local file exists, load the pretrained weight from local
      state_dict = torch.load(local_path, map_location="cpu")
      print(f'load the pretrained weight from local path: {local_path}')
    else:
      # if the local file does not exist, download the pretrained weight from network
      url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
      try:
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        print(f'download the pretrained weight from network')
      except Exception as e:
        print(f'download the pretrained weight from network failed: {e}')
        print(f'please download the model file to: {local_path}')
        raise e
    
    model.load_state_dict(state_dict, strict=True)
  print('model defined.')
  return model


def load_ResNet50(no_pretrain=False):
  from torchvision.models.resnet import resnet50
  pretrained = not no_pretrain
  model = resnet50(pretrained=pretrained)
  model.fc = torch.nn.Identity()
  print('model defined.')
  return model

def load_ResNet50_dino(no_pretrain=False):
  from torchvision.models.resnet import resnet50
  model = resnet50(pretrained=False)
  model.fc = torch.nn.Identity()
  if not no_pretrain:
    # check if the local path exists
    local_path = "./checkpoints/pretrained_models/dino_resnet50_pretrain.pth"
    import os
    if os.path.exists(local_path):
      # if the local file exists, load the pretrained weight from local
      state_dict = torch.load(local_path, map_location="cpu")
      print(f'load the pretrained weight from local path: {local_path}')
    else:
      # if the local file does not exist, download the pretrained weight from network
      try:
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/dino_resnet50_pretrain/dino_resnet50_pretrain.pth", map_location="cpu")
        print(f'download the pretrained weight from network')
      except Exception as e:
        print(f'download the pretrained weight from network failed: {e}')
        print(f'please download the model file to: {local_path}')
        raise e
    
    model.load_state_dict(state_dict, strict=False)
  return model

def load_ResNet50_clip(no_pretrain=False):
  from models import clip
  model, _ = clip.load('RN50', 'cpu')
  return model


def get_model(backbone='vit_small', classifier='protonet', args=None, styleAdv=False):
  if(backbone=='vit_small' and classifier == 'protonet'):
    extractor = load_ViTsmall()
    if(not styleAdv):
      #from models.pmf_protonet import ProtoNet
      from methods.protonet import ProtoNet
      model = ProtoNet(extractor)
    else:
      #from models.pmf_protonet_metatrain_vit_protonet_20221102 import ProtoNet
      #from methods.pmf_protonet_metatrain_vit_protonet_20221102 import ProtoNet
      from methods.StyleAdv_ViT_protonet import ProtoNet
      model = ProtoNet(extractor)

  if(backbone=='resnet50' and classifier == 'protonet'):
    extractor = load_ResNet50_dino()
    model = ProtoNet(extractor)

  if(backbone=='vit_small' and classifier == 'gnnnet'):
    extractor = load_ViTsmall()
    model = GnnNet(extractor, backbone_flag='vit_small', n_way = 5, n_support = args.nSupport)

  if(backbone=='resnet50' and classifier == 'gnnnet'):
    extractor = load_ResNet50_dino()
    model = GnnNet(extractor, backbone_flag='resnet50', n_way = 5, n_support = args.nSupport)
  return model




if __name__ == '__main__':
  input = torch.randn(16, 3, 224, 224)
  print('input:', input.size())
  model = load_ViTsmall()
  out = model(input)
  print('out:', out.size())

