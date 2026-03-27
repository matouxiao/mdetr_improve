#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从训练日志中提取loss并绘制loss曲线
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import os

# 尝试设置中文字体，如果失败则使用英文
try:
    # 尝试使用系统中文字体
    font_list = ['WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'STHeiti', 'Arial Unicode MS']
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    for font in font_list:
        if font in available_fonts:
            matplotlib.rcParams['font.sans-serif'] = [font]
            break
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
except:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def parse_log_file(log_file_path):
    """解析日志文件，提取loss数据"""
    losses = {
        'train_loss': [],
        'train_loss_ce': [],
        'train_loss_bbox': [],
        'train_loss_giou': [],
        'train_loss_contrastive_align': [],
        'epoch': []
    }
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                epoch = data.get('epoch', 0)
                losses['epoch'].append(epoch)
                losses['train_loss'].append(data.get('train_loss', 0))
                losses['train_loss_ce'].append(data.get('train_loss_ce', 0))
                losses['train_loss_bbox'].append(data.get('train_loss_bbox', 0))
                losses['train_loss_giou'].append(data.get('train_loss_giou', 0))
                losses['train_loss_contrastive_align'].append(data.get('train_loss_contrastive_align', 0))
            except json.JSONDecodeError as e:
                print(f"警告: 解析失败: {e}")
                continue
    
    return losses

def plot_loss_curves(losses, output_path='loss_curves.png'):
    """绘制loss曲线"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Loss Curves', fontsize=16, fontweight='bold')
    
    # 总loss
    ax = axes[0, 0]
    ax.plot(losses['epoch'], losses['train_loss'], 'b-', linewidth=1.5, label='Total Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('Total Loss (train_loss)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 交叉熵loss
    ax = axes[0, 1]
    ax.plot(losses['epoch'], losses['train_loss_ce'], 'r-', linewidth=1.5, label='CE Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('Cross-Entropy Loss (train_loss_ce)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 边界框loss
    ax = axes[0, 2]
    ax.plot(losses['epoch'], losses['train_loss_bbox'], 'g-', linewidth=1.5, label='Bbox Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('Bounding Box Loss (train_loss_bbox)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # GIoU loss
    ax = axes[1, 0]
    ax.plot(losses['epoch'], losses['train_loss_giou'], 'm-', linewidth=1.5, label='GIoU Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('GIoU Loss (train_loss_giou)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 对比对齐loss
    ax = axes[1, 1]
    ax.plot(losses['epoch'], losses['train_loss_contrastive_align'], 'c-', linewidth=1.5, label='Contrastive Align Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('Contrastive Align Loss (train_loss_contrastive_align)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 所有loss叠加图
    ax = axes[1, 2]
    ax.plot(losses['epoch'], losses['train_loss'], 'b-', linewidth=1.5, label='Total Loss', alpha=0.7)
    ax.plot(losses['epoch'], losses['train_loss_ce'], 'r-', linewidth=1.5, label='CE Loss', alpha=0.7)
    ax.plot(losses['epoch'], losses['train_loss_bbox'], 'g-', linewidth=1.5, label='Bbox Loss', alpha=0.7)
    ax.plot(losses['epoch'], losses['train_loss_giou'], 'm-', linewidth=1.5, label='GIoU Loss', alpha=0.7)
    ax.plot(losses['epoch'], losses['train_loss_contrastive_align'], 'c-', linewidth=1.5, label='Contrastive Align Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss Value')
    ax.set_title('All Losses Comparison')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Loss曲线已保存到: {output_path}")
    plt.close()

def main():
    log_file = '/workapp1219/detr/mdetr/refexp_results/swin_tiny_refcoco_ft/log.txt'
    output_file = '/workapp1219/detr/mdetr/refexp_results/swin_tiny_refcoco_ft/loss_curves.png'
    
    print("正在解析日志文件...")
    losses = parse_log_file(log_file)
    
    print(f"成功解析 {len(losses['epoch'])} 条记录")
    print(f"Epoch范围: {min(losses['epoch'])} - {max(losses['epoch'])}")
    print(f"总Loss范围: {min(losses['train_loss']):.4f} - {max(losses['train_loss']):.4f}")
    print(f"交叉熵Loss范围: {min(losses['train_loss_ce']):.4f} - {max(losses['train_loss_ce']):.4f}")
    print(f"边界框Loss范围: {min(losses['train_loss_bbox']):.4f} - {max(losses['train_loss_bbox']):.4f}")
    
    print("正在绘制Loss曲线...")
    plot_loss_curves(losses, output_file)
    print("完成！")

if __name__ == '__main__':
    main()
