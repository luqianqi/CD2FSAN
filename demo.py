import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import matplotlib.pyplot as plt
import clip

def get_tokenizer(clip_model):
    # 使用clip库中的tokenize函数
    return clip.tokenize

# 假设以下模块已经定义或导入
from model.model import Model
from utils.checkpoint import load_pretrain
#from utils.utils import get_tokenizer  # 假设存在tokenizer函数
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_pretrain(model, args, rank):
    if os.path.isfile(args.pretrain):
        checkpoint = torch.load(args.pretrain)
        pretrained_dict = checkpoint['state_dict']
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
        else:
            model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        assert (len([k for k, v in pretrained_dict.items()]) != 0)
        model_dict.update(pretrained_dict)
        if hasattr(model, 'module'):
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)
        print("=> loaded pretrain model at {}"
              .format(args.pretrain))
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
    else:
        print(("=> no pretrained file found at '{}'".format(args.pretrain)))
    return model

def get_args():
    parser = argparse.ArgumentParser(description='Single Image Inference for Referring Expression Segmentation')
    
    # 基本参数
    parser.add_argument('--gpu', default='0', help='GPU ID')
    parser.add_argument('--clip_model', default='ViT-B/16', type=str, help='CLIP model type')
    parser.add_argument('--size', default=480, type=int, help='Image size for processing')
    parser.add_argument('--seg_thresh', default=0.35, type=float, help='Threshold for segmentation mask')
    
    # 推理相关参数
    parser.add_argument('--single_inference', action='store_true', help='Enable single image inference mode')
    parser.add_argument('--image_path', type=str, default='', help='Path to input image')
    parser.add_argument('--text_query', type=str, default='', help='Text query for referring expression')
    parser.add_argument('--output_path', type=str, default='output_mask.png', help='Path to save the output mask')
    parser.add_argument('--pretrain', type=str, required=True, help='Path to pre-trained model checkpoint')
    
    args = parser.parse_args()
    return args

def inference_single_image(image_path, text_query, args, model, device):
    """
    Perform inference on a single image with a text query.
    """
    # 图片预处理
    input_transform = Compose([
        ToTensor(),
        Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], 
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    
    # 加载图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # 保存原始尺寸
    image = image.resize((args.size, args.size))
    image_tensor = input_transform(image).unsqueeze(0).to(device)
    
    # 文本预处理
    text_tokens = clip.tokenize(text_query, 17, truncate=True).to(device)
    word_mask = ~(text_tokens == 0).to(device)
    
    # 清理显存缓存
    torch.cuda.empty_cache()
    
    # 模型推理
    model.eval()
    with torch.no_grad():
        with torch.cuda.amp.autocast():  # 使用混合精度
            mask_out = model(image_tensor, text_tokens, word_mask)
            mask_out = mask_out.sigmoid()
    
    # 后处理
    mask_out = mask_out[0, 0].data.cpu().numpy()  # 提取第一个样本和第一个通道
    

    mask_out = mask_out.astype(np.float64)
    mask_out = cv2.UMat(mask_out)
    mask_out = cv2.resize(mask_out, original_size, interpolation=cv2.INTER_NEAREST)
    mask_out = mask_out.get()
    
    # 调整mask到原始图片尺寸
    #mask_out = cv2.resize(mask_out, original_size, interpolation=cv2.INTER_NEAREST)
    
    
    # 应用阈值
    pred_mask = (mask_out > args.seg_thresh).astype(np.uint8) * 255
    
    return pred_mask

def main(args):
    # 设置GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = Model(clip_model=args.clip_model, tunelang=True, num_query=16, fusion_dim=768)
    model = model.to(device)
    
    # 加载预训练权重
    if args.pretrain and os.path.isfile(args.pretrain):
        model = load_pretrain(model, args, 0)
        print(f"Loaded pre-trained model from {args.pretrain}")
    else:
        print("No pre-trained model found. Please provide a valid checkpoint path.")
        return
    
    # 推理
    if args.single_inference:
        if not os.path.exists(args.image_path):
            print(f"Image path {args.image_path} does not exist.")
            return
        
        mask = inference_single_image(
            args.image_path, 
            args.text_query, 
            args, 
            model, 
            device
        )
        
        # 保存结果
        cv2.imwrite(args.output_path, mask)
        print(f"Segmentation mask saved to {args.output_path}")
        
        # 可视化结果
        visualize_result(args.image_path, mask, args.output_path)

def visualize_result(image_path, mask, output_path):
    """
    可视化结果：将mask叠加到原始图片上
    """
    image = cv2.imread(image_path)
    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    mask_color[:, :, 1] = 0  # 只保留红色通道
    
    # 将mask叠加到图片上
    overlay = cv2.addWeighted(image, 0.7, mask_color, 0.3, 0)
    
    # 保存可视化结果
    cv2.imwrite(output_path.replace('.png', '_overlay.png'), overlay)
    print(f"Visualization saved to {output_path.replace('.png', '_overlay.png')}")

if __name__ == "__main__":
    args = get_args()
    main(args)
