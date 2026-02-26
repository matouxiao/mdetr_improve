"""使用 RefCOCO 数据集进行指代表达理解（Referring Expression Comprehension）任务推理测试"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
import json
from pathlib import Path
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from util.misc import NestedTensor
from datasets.refexp import build as build_refexp

# 加载模型
print("正在加载模型...")
model, postprocessor = torch.hub.load(
    '/workapp1219/detr/mdetr',
    'mdetr_resnet101',
    pretrained=True,
    return_postprocessor=True,
    source='local'
)
model.eval()
print("模型加载完成！")

# 预处理
transform = Compose([
    Resize((800, 1333)),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def run_inference(img, caption, device="cuda"):
    """对单张图片进行推理"""
    h, w = img.height, img.width
    img_tensor = transform(img).unsqueeze(0)
    samples = NestedTensor.from_tensor_list([img_tensor.squeeze(0)])
    
    model.to(device)
    samples = samples.to(device)
    
    with torch.no_grad():
        # 第一次：编码
        memory_cache = model(samples, captions=[caption], encode_and_save=True)
        # 第二次：解码得到预测
        outputs = model(samples, captions=[caption], encode_and_save=False, memory_cache=memory_cache)
       
    target_sizes = torch.tensor([[h, w]], device=device)
    results = postprocessor(outputs, target_sizes)
    return results[0]

# 加载 RefCOCO 数据集（指代表达理解任务）
print("\n正在加载 RefCOCO 数据集（指代表达理解任务）...")
class Args:
    def __init__(self):
        self.coco_path = "/workapp1219/detr/mdetr/coco"
        self.refexp_ann_path = "mdetr_annotations/"
        self.refexp_dataset_name = "refcoco"  # 可选: "refcoco", "refcoco+", "refcocog"
        self.masks = False
        self.text_encoder_type = "roberta-base"
        self.test = False
        self.test_type = None

args = Args()

try:
    dataset = build_refexp("val", args)  # 使用验证集
    print(f"RefCOCO 数据集加载完成！共有 {len(dataset)} 条指代表达")
    print(f"任务类型: 指代表达理解（Referring Expression Comprehension）")
    print(f"数据集: {args.refexp_dataset_name}")
except Exception as e:
    print(f"加载 RefCOCO 数据集失败: {e}")
    print("尝试使用 COCO 数据集作为备选...")
    from datasets.coco import build as build_coco
    dataset = build_coco("train", args)
    print(f"COCO 数据集加载完成！共有 {len(dataset)} 张图片")
    print("注意：这是目标检测任务，不是指代表达理解任务")

# 测试前 100 条数据
num_test = min(100, len(dataset))
print(f"\n开始测试前 {num_test} 条数据...")
print("=" * 80)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

# 创建输出目录
output_dir = Path("/workapp1219/detr/mdetr/inference_results")
output_dir.mkdir(exist_ok=True)
print(f"预测结果将保存到: {output_dir}")

# 存储所有预测结果
all_predictions = []

total_boxes = 0
success_count = 0
error_count = 0

for idx in range(num_test):
    try:
        # 获取数据
        img, target = dataset[idx]
        
        # 处理 image_id（可能是 tensor 或 int）
        image_id = target['image_id']
        if torch.is_tensor(image_id):
            image_id = image_id.item()
        image_id = int(image_id)
        
        # 获取图片路径
        img_info = dataset.coco.loadImgs(image_id)[0]
        img_filename = img_info['file_name']
        img_path = Path("/workapp1219/detr/mdetr/coco/train2014") / img_filename
        
        # 检查图片文件是否存在
        if not img_path.exists():
            raise FileNotFoundError(f"图片文件不存在: {img_path}")
        
        # 获取指代表达（referring expression）
        # RefCOCO 数据集中，caption 字段包含指代表达
        if 'caption' in target:
            caption = target['caption']  # 指代表达，例如："the person on the left"
        else:
            # 如果没有 caption，尝试从数据集获取
            try:
                img_info = dataset.coco.loadImgs(image_id)[0]
                caption = img_info.get('caption', 'objects in image')
            except:
                caption = "objects in image"
        
        # 加载原始图片进行推理
        img_pil = Image.open(img_path).convert("RGB")
        
        # 执行推理
        results = run_inference(img_pil, caption, device=device)
        
        # 获取检测框和置信度
        boxes = results["boxes"].cpu() if torch.is_tensor(results["boxes"]) else torch.tensor(results["boxes"])
        scores = results["scores"].cpu() if torch.is_tensor(results["scores"]) else torch.tensor(results["scores"])
        labels = results["labels"].cpu() if torch.is_tensor(results["labels"]) else torch.tensor(results["labels"])
        
        # 根据置信度阈值过滤检测框（只保留高置信度的）
        confidence_threshold = 0.5  # 置信度阈值，可以调整
        keep = scores > confidence_threshold
        filtered_boxes = boxes[keep]
        filtered_scores = scores[keep]
        filtered_labels = labels[keep]
        
        num_boxes_all = len(boxes)  # 所有检测框数量（包括低置信度的）
        num_boxes_filtered = len(filtered_boxes)  # 过滤后的检测框数量
        total_boxes += num_boxes_filtered  # 只统计高置信度的框
        
        # 保存预测结果（包含过滤前后的信息）
        prediction_data = {
            "image_id": image_id,
            "image_filename": img_filename,
            "referring_expression": caption,  # 指代表达
            "task_type": "Referring Expression Comprehension",  # 任务类型
            "num_boxes_all": num_boxes_all,  # 所有检测框（通常是100个，因为num_queries=100）
            "num_boxes_filtered": num_boxes_filtered,  # 过滤后的检测框
            "confidence_threshold": confidence_threshold,
            "boxes": filtered_boxes.tolist(),
            "scores": filtered_scores.tolist(),
            "labels": filtered_labels.tolist(),
            # 也保存所有检测框的统计信息
            "all_scores_stats": {
                "min": float(scores.min().item()),
                "max": float(scores.max().item()),
                "mean": float(scores.mean().item()),
                "median": float(scores.median().item()),
            },
            # 指代表达理解任务的评估指标
            "is_single_detection": num_boxes_filtered == 1,  # 是否只检测到一个对象（理想情况）
        }
        all_predictions.append(prediction_data)
        
        # 显示结果
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"[{idx+1}/{num_test}] 图片 ID: {image_id}, "
                  f"所有检测框: {num_boxes_all}, "
                  f"高置信度框(>{confidence_threshold}): {num_boxes_filtered}, "
                  f"指代表达: {caption[:60]}...")
            if num_boxes_filtered > 0:
                top_scores = filtered_scores[:3].tolist()
                print(f"  前 3 个高置信度框: {[f'{s:.3f}' for s in top_scores]}")
                # 对于指代表达理解任务，通常期望只有一个高置信度的检测框
                if num_boxes_filtered == 1:
                    print(f"  ✓ 成功检测到唯一对象（置信度: {filtered_scores[0].item():.3f}）")
            else:
                print(f"  警告: 没有检测到高置信度的对象（最高置信度: {scores.max().item():.3f}）")
        
        success_count += 1
        
    except Exception as e:
        error_count += 1
        import traceback
        error_msg = f"[{idx+1}/{num_test}] 错误: {type(e).__name__}: {str(e)}"
        print(error_msg)
        # 只在第一个错误时显示完整 traceback
        if error_count == 1:
            print("详细错误信息:")
            traceback.print_exc()
        if error_count > 5:  # 如果错误太多，停止
            print("错误过多，停止测试")
            break

# 保存预测结果到 JSON 文件
results_file = output_dir / "predictions.json"
print(f"\n正在保存预测结果到 {results_file}...")
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump({
        "summary": {
            "total_tested": num_test,
            "success": success_count,
            "failed": error_count,
            "avg_boxes": total_boxes / success_count if success_count > 0 else 0
        },
        "predictions": all_predictions
    }, f, indent=2, ensure_ascii=False)
print(f"✓ 预测结果已保存到: {results_file}")

# 统计指代表达理解任务的指标
single_detection_count = sum(1 for p in all_predictions if p.get("is_single_detection", False))

# 保存简化的统计信息
summary_file = output_dir / "summary.txt"
with open(summary_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("指代表达理解（Referring Expression Comprehension）任务测试结果\n")
    f.write("=" * 80 + "\n")
    f.write(f"任务类型: 指代表达理解（Referring Expression Comprehension）\n")
    f.write(f"数据集: RefCOCO\n")
    f.write(f"测试总数: {num_test}\n")
    f.write(f"成功: {success_count}\n")
    f.write(f"失败: {error_count}\n")
    if success_count > 0:
        avg_boxes = total_boxes / success_count
        f.write(f"平均检测框数量: {avg_boxes:.2f}\n")
        f.write(f"唯一检测成功数（理想情况）: {single_detection_count} ({single_detection_count/success_count*100:.1f}%)\n")
    f.write("\n详细结果请查看 predictions.json 文件\n")
    f.write("=" * 80 + "\n")
print(f"✓ 统计信息已保存到: {summary_file}")

# 统计结果
print("\n" + "=" * 80)
print("指代表达理解任务测试完成！")
print(f"任务类型: Referring Expression Comprehension")
print(f"成功: {success_count}/{num_test}")
print(f"失败: {error_count}/{num_test}")
if success_count > 0:
    avg_boxes = total_boxes / success_count
    print(f"平均检测框数量: {avg_boxes:.2f}")
    single_detection_count = sum(1 for p in all_predictions if p.get("is_single_detection", False))
    print(f"唯一检测成功数（理想情况）: {single_detection_count}/{success_count} ({single_detection_count/success_count*100:.1f}%)")
print("\n预测结果保存位置:")
print(f"  - 详细结果 (JSON): {results_file}")
print(f"  - 统计信息 (TXT): {summary_file}")
print("=" * 80)