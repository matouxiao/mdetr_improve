"""可视化预测结果：在图片上绘制检测框"""
import json
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import torch

def draw_boxes_on_image(image_path, boxes, scores, labels=None, caption="", output_path=None):
    """
    在图片上绘制检测框
    
    Args:
        image_path: 图片路径
        boxes: 检测框列表 [[x0, y0, x1, y1], ...]
        scores: 置信度列表
        labels: 标签列表（可选）
        caption: 指代表达或描述文本
        output_path: 输出图片路径
    """
    # 加载图片
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    
    # 尝试加载字体
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except:
        try:
            font = ImageFont.truetype("arial.ttf", 16)
            font_small = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
            font_small = ImageFont.load_default()
    
    # 绘制检测框
    colors = [
        (255, 0, 0),    # 红色
        (0, 255, 0),    # 绿色
        (0, 0, 255),    # 蓝色
        (255, 255, 0),  # 黄色
        (255, 0, 255),  # 洋红
        (0, 255, 255),  # 青色
    ]
    
    for i, (box, score) in enumerate(zip(boxes, scores)):
        x0, y0, x1, y1 = box
        color = colors[i % len(colors)]
        
        # 绘制矩形框
        draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
        
        # 绘制置信度标签
        label_text = f"{score:.2f}"
        if labels is not None and i < len(labels):
            label_text = f"Label {labels[i]}: {score:.2f}"
        
        # 计算文本位置（在框的上方）
        bbox = draw.textbbox((0, 0), label_text, font=font_small)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 绘制文本背景
        text_x = x0
        text_y = max(0, y0 - text_height - 4)
        draw.rectangle(
            [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
            fill=color
        )
        
        # 绘制文本
        draw.text((text_x + 2, text_y + 2), label_text, fill=(255, 255, 255), font=font_small)
    
    # 在图片顶部绘制指代表达
    if caption:
        # 计算文本大小
        bbox = draw.textbbox((0, 0), caption, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 绘制半透明背景
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [0, 0, img.width, text_height + 20],
            fill=(0, 0, 0, 180)  # 半透明黑色
        )
        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)
        
        # 绘制文本
        text_x = (img.width - text_width) // 2
        text_y = 10
        draw.text((text_x, text_y), caption, fill=(255, 255, 255), font=font)
    
    # 保存图片
    if output_path:
        img.save(output_path)
        print(f"已保存可视化结果到: {output_path}")
    
    return img

def visualize_predictions(predictions_file, output_dir=None, num_images=None, start_idx=0):
    """
    可视化预测结果
    
    Args:
        predictions_file: 预测结果JSON文件路径
        output_dir: 输出目录
        num_images: 要可视化的图片数量（None表示全部）
        start_idx: 起始索引
    """
    # 读取预测结果
    print(f"正在读取预测结果: {predictions_file}")
    with open(predictions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    predictions = data.get('predictions', [])
    summary = data.get('summary', {})
    
    print(f"找到 {len(predictions)} 条预测结果")
    print(f"成功: {summary.get('success', 0)}, 失败: {summary.get('failed', 0)}")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = Path(predictions_file).parent / "visualizations"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"可视化结果将保存到: {output_dir}")
    
    # 确定要可视化的图片范围
    end_idx = start_idx + num_images if num_images else len(predictions)
    end_idx = min(end_idx, len(predictions))
    
    print(f"\n开始可视化图片 {start_idx} 到 {end_idx-1}...")
    
    # 可视化每张图片
    for idx in range(start_idx, end_idx):
        pred = predictions[idx]
        
        try:
            image_id = pred['image_id']
            image_filename = pred.get('image_filename', f"COCO_train2014_{image_id:012d}.jpg")
            referring_expression = pred.get('referring_expression', pred.get('caption', ''))
            
            # 构建图片路径
            image_path = Path("/workapp1219/detr/mdetr/coco/train2014") / image_filename
            if not image_path.exists():
                print(f"警告: 图片不存在 {image_path}，跳过")
                continue
            
            # 获取检测框和置信度
            boxes = pred.get('boxes', [])
            scores = pred.get('scores', [])
            num_boxes = pred.get('num_boxes_filtered', len(boxes))
            
            if num_boxes == 0:
                print(f"[{idx+1}/{end_idx-start_idx}] 图片 {image_id}: 没有检测框，跳过")
                continue
            
            # 输出路径
            output_path = output_dir / f"{image_id:06d}_{idx:04d}_visualization.jpg"
            
            # 绘制检测框
            img = draw_boxes_on_image(
                image_path,
                boxes,
                scores,
                labels=pred.get('labels', None),
                caption=referring_expression[:80],  # 限制长度
                output_path=output_path
            )
            
            print(f"[{idx+1}/{end_idx-start_idx}] 图片 {image_id}: {num_boxes} 个检测框 -> {output_path.name}")
            
        except Exception as e:
            print(f"[{idx+1}/{end_idx-start_idx}] 错误: {str(e)}")
            continue
    
    print(f"\n可视化完成！结果保存在: {output_dir}")
    print(f"共处理 {end_idx - start_idx} 张图片")

def main():
    parser = argparse.ArgumentParser(description="可视化MDETR预测结果")
    parser.add_argument(
        "--predictions_file",
        type=str,
        default="/workapp1219/detr/mdetr/inference_results/predictions.json",
        help="预测结果JSON文件路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录（默认：predictions_file所在目录/visualizations）"
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=20,
        help="要可视化的图片数量（默认：20）"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="起始索引（默认：0）"
    )
    
    args = parser.parse_args()
    
    visualize_predictions(
        predictions_file=args.predictions_file,
        output_dir=args.output_dir,
        num_images=args.num_images,
        start_idx=args.start_idx
    )

if __name__ == "__main__":
    main()
