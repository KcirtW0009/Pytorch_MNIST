import torch
import torch.nn as nn
from CNN import CNN
from pytorch_mnist import download_and_preprocess_mnist
import numpy as np
from collections import defaultdict
import os
import cv2


def load_model(model_path: str, device: torch.device):
    """
    加载训练好的CNN模型

    Args:
        model_path (str): 模型文件路径
        device (torch.device): 运行设备

    Returns:
        model: 加载好权重的CNN模型
        info (dict): 模型信息字典
    """
    print(f"正在加载模型: {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model = CNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    info = {
        'epoch': checkpoint.get('epoch', '未知'),
        'test_accuracy': checkpoint.get('test_accuracy', '未知'),
        'test_loss': checkpoint.get('test_loss', '未知'),
        'architecture': checkpoint.get('model_architecture', '未知')
    }

    print(f"✅ 模型加载成功！")
    print(f"   - 训练轮数: {info['epoch']}")
    print(f"   - 原始测试准确率: {info['test_accuracy']}%")
    print(f"   - 设备: {device}")
    print()

    return model, info


def test_model(model, test_loader, device: torch.device,
               max_samples: int = 30, max_error_samples: int = 20):
    """
    对测试集进行评估并收集详细结果

    Args:
        model: 已加载的模型
        test_loader: 测试集数据加载器
        device: 运行设备
        max_samples (int): 最大收集的顺序样本数量（用于顺序展示）
        max_error_samples (int): 最大收集的错误样本数量（用于错误专展）

    Returns:
        dict: 包含所有测试结果的字典
    """
    correct = 0
    total = 0

    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_names = [str(i) for i in range(10)]

    sequential_samples = []
    error_samples = []

    print("开始测试...")
    print("-" * 60)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                confidence = torch.softmax(outputs[i], dim=0)[pred].item()
                is_correct = (pred == label)

                class_total[label] += 1
                if is_correct:
                    class_correct[label] += 1
                else:
                    if len(error_samples) < max_error_samples:
                        error_samples.append({
                            'image': images[i].cpu(),
                            'true_label': label,
                            'predicted_label': pred,
                            'confidence': confidence
                        })

                if len(sequential_samples) < max_samples:
                    sequential_samples.append({
                        'image': images[i].cpu(),
                        'true_label': label,
                        'predicted_label': pred,
                        'confidence': confidence,
                        'is_correct': is_correct
                    })

            if (batch_idx + 1) % 50 == 0:
                current_acc = 100.0 * correct / total
                print(f"   批次 [{batch_idx+1}/{len(test_loader)}] - 当前准确率: {current_acc:.2f}%")

    overall_accuracy = 100.0 * correct / total

    class_accuracies = {}
    for i in range(10):
        if class_total[i] > 0:
            acc = 100.0 * class_correct[i] / class_total[i]
            class_accuracies[i] = {
                'correct': class_correct[i],
                'total': class_total[i],
                'accuracy': acc
            }
        else:
            class_accuracies[i] = {'correct': 0, 'total': 0, 'accuracy': 0.0}

    results = {
        'overall_accuracy': overall_accuracy,
        'correct': correct,
        'total': total,
        'class_accuracies': class_accuracies,
        'sequential_samples': sequential_samples,
        'error_samples': error_samples
    }

    return results


def print_results(results: dict):
    """
    打印详细的测试结果

    Args:
        results (dict): test_model返回的结果字典
    """
    print("\n" + "=" * 70)
    print("📊 测试结果报告")
    print("=" * 70)

    print("\n【整体性能】")
    print(f"   总样本数: {results['total']}")
    print(f"   正确预测: {results['correct']}")
    print(f"   错误预测: {results['total'] - results['correct']}")
    print(f"   🎯 整体准确率: {results['overall_accuracy']:.2f}%")

    print("\n【各类别准确率】")
    print("-" * 50)
    print(f"{'类别':^8} | {'正确/总数':^12} | {'准确率':^10} | {'状态'}")
    print("-" * 50)

    for class_id in range(10):
        acc_info = results['class_accuracies'][class_id]
        correct = acc_info['correct']
        total = acc_info['total']
        accuracy = acc_info['accuracy']

        status = "✅" if accuracy >= 95 else ("⚠️" if accuracy >= 90 else "❌")

        print(f"{class_id:^8} | {correct}/{total:^8} | {accuracy:^9.2f}% | {status}")

    avg_class_acc = np.mean([results['class_accuracies'][i]['accuracy'] for i in range(10)])
    print("-" * 50)
    print(f"{'平均':^8} | {'':^12} | {avg_class_acc:^9.2f}% |")

    error_count = len(results['error_samples'])
    print(f"\n【错误样本统计】")
    print(f"   收集到 {error_count} 个错误预测样本")

    if error_count > 0:
        print("\n【错误样本详情（前20个）】")
        print("-" * 80)
        for idx, sample in enumerate(results['error_samples'], 1):
            true_label = sample['true_label']
            pred_label = sample['predicted_label']
            conf = sample['confidence'] * 100
            print(f"   样本 {idx:>3}: 真实标签={true_label}, 预测标签={pred_label}, 置信度={conf:.2f}%")

    print("\n" + "=" * 70)


def tensor_to_image(tensor_image, scale: int = 10):
    """
    将PyTorch Tensor转换为OpenCV图像格式

    Args:
        tensor_image: PyTorch张量，形状为 [1, 28, 28] 或 [28, 28]
        scale (int): 图像放大倍数（默认10倍，从28x28放大到280x280）

    Returns:
        numpy.ndarray: OpenCV格式的灰度图像
    """
    if isinstance(tensor_image, torch.Tensor):
        image = tensor_image.numpy()
    else:
        image = np.array(tensor_image)

    if len(image.shape) == 3:
        image = image.squeeze()

    image = ((image * 0.3081 + 0.1307) * 255).clip(0, 255).astype(np.uint8)

    image = cv2.resize(image, (28 * scale, 28 * scale), interpolation=cv2.INTER_NEAREST)

    return image


def draw_prediction_on_image(image, true_label: int, predicted_label: int,
                             confidence: float, is_correct: bool = True):
    """
    在图像上绘制预测结果标注

    Args:
        image: OpenCV格式的图像
        true_label (int): 真实标签
        predicted_label (int): 预测标签
        confidence (float): 预测置信度 (0-1)
        is_correct (bool): 预测是否正确

    Returns:
        numpy.ndarray: 带标注的图像
    """
    annotated = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    color = (0, 255, 0) if is_correct else (0, 0, 255)

    status_text = "✓ Correct" if is_correct else "✗ Wrong"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 35), color, -1)

    label_text = f"True: {true_label} | Pred: {predicted_label} | {status_text} | Conf: {confidence*100:.1f}%"
    cv2.putText(annotated, label_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    digit_text = f"Predicted: {predicted_label}"
    cv2.putText(annotated, digit_text, (10, annotated.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

    return annotated


def show_single_image(sample: dict, scale: int = 10):
    """
    显示单个测试图片及其预测结果

    Args:
        sample (dict): 包含图片和预测信息的字典
            - 'image': 图像tensor
            - 'true_label': 真实标签
            - 'predicted_label': 预测标签
            - 'confidence': 置信度
        scale (int): 图像放大倍数
    """
    image = tensor_to_image(sample['image'], scale)

    is_correct = sample['true_label'] == sample['predicted_label']
    annotated = draw_prediction_on_image(
        image,
        sample['true_label'],
        sample['predicted_label'],
        sample['confidence'],
        is_correct
    )

    window_title = f"Sample - True: {sample['true_label']} | Predicted: {sample['predicted_label']}"
    cv2.imshow(window_title, annotated)
    print(f"\n📷 显示单个测试图片")
    print(f"   按任意键关闭窗口...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_error_samples_grid(error_samples: list, cols: int = 4, scale: int = 8,
                            samples_per_page: int = 10):
    """
    分批展示错误预测样本（专用于错误分析）

    Args:
        error_samples (list): 错误样本列表
        cols (int): 每行显示的图片数量（默认4，适合窗口显示）
        scale (int): 单个图片的放大倍数（默认8倍，224x224）
        samples_per_page (int): 每页/每轮显示的错误样本数量（默认10个）
    """
    if not error_samples:
        print("\n⚠️ 没有错误样本可以展示")
        return

    n_samples = len(error_samples)

    total_pages = (n_samples + samples_per_page - 1) // samples_per_page

    print(f"\n📊 错误样本专展模式：共 {n_samples} 个错误样本")
    print(f"   分 {total_pages} 轮展示，每轮 {samples_per_page} 个样本")

    for page in range(total_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, n_samples)
        page_samples = error_samples[start_idx:end_idx]

        n_page_samples = len(page_samples)
        rows = (n_page_samples + cols - 1) // cols

        img_size = 28 * scale
        padding = 10

        grid_width = cols * img_size + (cols + 1) * padding
        grid_height = rows * img_size + (rows + 1) * padding + 50

        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 248

        title_text = f"Error Samples Only [{page+1}/{total_pages}] (Showing: {start_idx+1}-{end_idx}/{n_samples})"
        cv2.putText(grid, title_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 150), 2)

        subtitle_text = f"Detailed Error Analysis - Round {page+1}"
        cv2.putText(grid, subtitle_text, (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)

        for idx, sample in enumerate(page_samples):
            row = idx // cols
            col = idx % cols

            x_start = padding + col * (img_size + padding)
            y_start = padding + row * (img_size + padding) + 52

            image = tensor_to_image(sample['image'], scale)

            annotated = draw_prediction_on_image(
                image,
                sample['true_label'],
                sample['predicted_label'],
                sample['confidence'],
                is_correct=False
            )

            grid[y_start:y_start+img_size, x_start:x_start+img_size] = annotated

        window_title = f"Error Analysis - Round {page+1}/{total_pages}"
        cv2.imshow(window_title, grid)
        print(f"\n   � 第 {page+1}/{total_pages} 轮：展示错误样本 {start_idx+1}-{end_idx}")
        print(f"      窗口尺寸: {grid_width}x{grid_height} 像素")
        print(f"      按任意键继续下一轮（按 ESC 或 Q 可退出）...")

        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):
            print("\n   ⏹️ 用户中断展示")
            break

        try:
            cv2.destroyWindow(window_title)
        except Exception:
            pass

    cv2.destroyAllWindows()
    print(f"\n   ✅ 错误样本展示完成！共浏览了 {min(page+1, total_pages)} 轮")


def show_sequential_samples_grid(sequential_samples: list, cols: int = 4, scale: int = 8,
                                  samples_per_page: int = 12):
    """
    按测试集原始顺序分批展示混合样本（包含正确和错误的预测）

    Args:
        sequential_samples (list): 按顺序收集的样本列表
        cols (int): 每行显示的图片数量（默认4，适合窗口显示）
        scale (int): 单个图片的放大倍数（默认8倍，224x224）
        samples_per_page (int): 每页/每轮显示的样本数量（默认12个）
    """
    if not sequential_samples:
        print("\n⚠️ 没有样本可以展示")
        return

    n_samples = len(sequential_samples)
    n_correct_total = sum(1 for s in sequential_samples if s['is_correct'])
    n_wrong_total = n_samples - n_correct_total

    total_pages = (n_samples + samples_per_page - 1) // samples_per_page

    print(f"\n📋 顺序展示模式：共 {n_samples} 个样本（✅正确: {n_correct_total} | ❌错误: {n_wrong_total}）")
    print(f"   分 {total_pages} 轮展示，每轮 {samples_per_page} 个样本")

    for page in range(total_pages):
        start_idx = page * samples_per_page
        end_idx = min(start_idx + samples_per_page, n_samples)
        page_samples = sequential_samples[start_idx:end_idx]

        n_page_samples = len(page_samples)
        rows = (n_page_samples + cols - 1) // cols

        n_correct_page = sum(1 for s in page_samples if s['is_correct'])
        n_wrong_page = n_page_samples - n_correct_page

        img_size = 28 * scale
        padding = 10

        grid_width = cols * img_size + (cols + 1) * padding
        grid_height = rows * img_size + (rows + 1) * padding + 55

        grid = np.ones((grid_height, grid_width, 3), dtype=np.uint8) * 245

        title_text = f"Sequential Samples [{page+1}/{total_pages}] (Showing: {start_idx+1}-{end_idx}/{n_samples})"
        cv2.putText(grid, title_text, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

        stats_text = f"This Page: Correct={n_correct_page} | Wrong={n_wrong_page}"
        cv2.putText(grid, stats_text, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        legend_y = grid_height - 18
        cv2.rectangle(grid, (10, legend_y - 12), (30, legend_y), (0, 255, 0), -1)
        cv2.putText(grid, "= Correct", (35, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)

        cv2.rectangle(grid, (150, legend_y - 12), (170, legend_y), (0, 0, 255), -1)
        cv2.putText(grid, "= Wrong", (175, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)

        for idx, sample in enumerate(page_samples):
            row = idx // cols
            col = idx % cols

            x_start = padding + col * (img_size + padding)
            y_start = padding + row * (img_size + padding) + 55

            image = tensor_to_image(sample['image'], scale)

            annotated = draw_prediction_on_image(
                image,
                sample['true_label'],
                sample['predicted_label'],
                sample['confidence'],
                is_correct=sample['is_correct']
            )

            grid[y_start:y_start+img_size, x_start:x_start+img_size] = annotated

        window_title = f"Sequential View - Round {page+1}/{total_pages}"
        cv2.imshow(window_title, grid)
        print(f"\n   � 第 {page+1}/{total_pages} 轮：展示样本 {start_idx+1}-{end_idx}")
        print(f"      本轮统计：✅{n_correct_page} 正确 | ❌{n_wrong_page} 错误")
        print(f"      窗口尺寸: {grid_width}x{grid_height} 像素")
        print(f"      按任意键继续下一轮...")

        key = cv2.waitKey(0)
        if key == 27 or key == ord('q'):
            print("\n   ⏹️ 用户中断展示")
            break

        try:
            cv2.destroyWindow(window_title)
        except Exception:
            pass

    cv2.destroyAllWindows()
    print(f"\n   ✅ 顺序展示完成！共浏览了 {min(page+1, total_pages)} 轮")


def main():
    """
    主函数：执行完整的模型测试流程
    """
    print("\n" + "=" * 70)
    print("🔍 MNIST 手写数字识别 - 模型测试程序")
    print("=" * 70 + "\n")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}\n")

        model_path = './models/cnn_mnist_20260525_211841.pkl'

        model, model_info = load_model(model_path, device)

        print("正在加载测试数据集...")
        _, test_loader, _ = download_and_preprocess_mnist(
            data_root='./data',
            batch_size=64,
            num_workers=2,
            validation_split=0
        )
        print("✅ 测试数据集加载完成！\n")

        results = test_model(model, test_loader, device,
                            max_samples=48, max_error_samples=30)

        print_results(results)

        print("\n" + "=" * 70)
        print("🖼️  OpenCV 图像展示")
        print("=" * 70)

        if results['sequential_samples']:
            print("\n【Phase 1】Sequential Display - Mixed Correct & Error Samples (Multi-Round)...")
            show_sequential_samples_grid(
                results['sequential_samples'],
                cols=4,
                scale=8,
                samples_per_page=12
            )
        else:
            print("\n⚠️ 没有收集到顺序样本")

        if results['error_samples']:
            print("\n【Phase 2】Dedicated Error Samples View - Detailed Analysis (Multi-Round)...")
            show_error_samples_grid(
                results['error_samples'],
                cols=4,
                scale=8,
                samples_per_page=10
            )
        else:
            print("\n🎉 太棒了！测试集上没有错误预测的样本！")

        print("\n✅ 测试完成！")

    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("请确保模型文件存在，或检查文件路径是否正确")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
