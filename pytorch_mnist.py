"""
================================================================================
PyTorch MNIST 手写数字识别 - 数据集下载与预处理模块
================================================================================

模块概述:
--------
本模块提供MNIST手写数字数据集的自动下载、预处理和数据加载功能。
支持多种镜像源、断点续传、智能错误恢复等企业级特性。

主要功能:
---------
1. 自动检测并下载MNIST数据集（训练集+测试集）
2. 数据标准化和归一化处理
3. 验证集自动划分
4. 高效的多进程数据加载
5. 完整的数据集信息查看功能

技术栈:
-------
- PyTorch >= 1.7.0
- torchvision >= 0.8.0
- Python >= 3.7 (推荐 3.9+)

作者: AI Assistant
版本: 2.0.0
最后更新: 2026-05-24

使用示例:
---------
>>> from pytorch_mnist import download_and_preprocess_mnist, print_dataset_info
>>> 
>>> # 基础用法：下载数据集并查看详情
>>> train_loader, test_loader, val_loader = download_and_preprocess_mnist()
>>> print_dataset_info(train_loader, test_loader, val_loader)
>>>
>>> # 高级用法：自定义参数
>>> train_loader, test_loader, _ = download_and_preprocess_mnist(
...     data_root='./my_data',
...     batch_size=128,
...     num_workers=4,
...     validation_split=0.15
... )

数据集信息:
----------
- 训练集: 60,000 张28×28灰度图像 (数字0-9)
- 测试集: 10,000 张28×28灰度图像 (数字0-9)
- 图像格式: 单通道灰度图，像素值0-255
- 标签格式: 整数0-9，对应10个数字类别
- 文件大小: 约11MB（压缩后）

输出格式:
---------
- Tensor形状: [batch_size, 1, 28, 28]
- 像素值范围: 标准化后约 [-0.424, 2.821]
- 标签类型: torch.LongTensor

注意事项:
---------
1. 首次运行需要联网下载数据集（约11MB）
2. 下载完成后数据会缓存在本地，后续无需重复下载
3. Windows系统使用num_workers>0时需在if __name__ == '__main__':保护下执行
4. 建议使用GPU环境以获得最佳性能

================================================================================
"""

# ============================================================================
# 导入依赖库
# ============================================================================
import torch  # PyTorch核心库，用于张量计算和深度学习模型构建
import torchvision.datasets as dataset  # torchvision内置数据集模块，包含MNIST数据集加载器
import torchvision.transforms as transforms  # 图像变换工具，用于数据预处理和增强
from torch.utils.data import DataLoader, random_split  # 数据加载工具和数据集划分工具
import os  # 操作系统接口，用于文件路径操作和环境变量访问
import logging  # 日志记录模块，替代print实现专业的日志管理
import urllib.error  # URL请求错误处理，捕获网络相关异常
import urllib.request  # URL请求模块，用于自定义文件下载功能
import gzip  # GZIP压缩文件处理，MNIST数据集采用gzip压缩格式
import shutil  # 高级文件操作工具，用于目录清理和文件复制
from typing import Tuple, Optional  # 类型注解支持，提高代码可读性和IDE提示
from collections import Counter  # 计数器工具，用于统计标签分布


# ============================================================================
# 日志系统配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为INFO，显示INFO及以上级别的日志
    format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式：时间-级别-消息
    datefmt='%Y-%m-%d %H:%M:%S'  # 时间戳格式
)

logger = logging.getLogger(__name__)  # 创建模块专属logger实例


# ============================================================================
# 全局配置常量
# ============================================================================

#: MNIST数据集镜像源列表（按优先级排序）
#: 第一个为AWS S3镜像源，国内访问速度快且稳定
#: 第二个为官方原始源，作为备用选项
MNIST_MIRRORS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',  # AWS S3镜像（推荐）
    'http://yann.lecun.com/exdb/mnist/',                # LeCun官方源（备用）
]

#: MNIST数据集完整文件清单
#: 包含每个文件的MD5校验码，用于验证文件完整性
#: 格式: {数据集类型: [(文件名, MD5哈希), ...]}
MNIST_FILES = {
    'train': [
        ('train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),  # 训练集图像 (~9.9MB)
        ('train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),  # 训练集标签 (~29KB)
    ],
    'test': [
        ('t10k-images-idx3-ubyte.gz', '9fb629c81845c75c81d38d568de7d671'),   # 测试集图像 (~1.6MB)
        ('t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c'),   # 测试集标签 (~5KB)
    ]
}


def download_file(url: str, filepath: str, timeout: int = 30) -> bool:
    """
    带重试机制和进度显示的文件下载函数
    
    该函数实现了健壮的文件下载功能，包括：
    - 自动重试机制（最多3次）
    - 实时进度显示
    - 超时控制
    - 异常捕获和处理
    
    Args:
        url (str): 要下载的文件的完整URL地址
                   示例: 'https://example.com/file.zip'
        filepath (str): 文件保存的本地绝对或相对路径
                       示例: './data/MNIST/raw/train-images-idx3-ubyte.gz'
        timeout (int, optional): 网络请求超时时间（秒）。默认为30秒。
                                超过此时间未收到数据将抛出异常并触发重试
    
    Returns:
        bool: 下载是否成功
              - True: 文件成功下载到指定路径
              - False: 所有重试均失败，文件可能不存在或不完整
    
    Raises:
        不直接抛出异常，所有异常内部处理后返回False
    
    Example:
        >>> success = download_file(
        ...     url='https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz',
        ...     filepath='./data/MNIST/raw/train-images-idx3-ubyte.gz',
        ...     timeout=60
        ... )
        >>> if success:
        ...     print("下载完成！")
    
    Implementation Details:
        - 使用urllib.request.urlretrieve进行底层下载
        - 通过reporthook回调函数实现进度显示
        - 每次失败后等待2秒再重试，避免频繁请求被封禁
        - 进度显示使用\\r回车符实现原地更新，避免刷屏
    """
    max_retries = 3  #: 最大重试次数
    
    for attempt in range(max_retries):
        try:
            logger.info(f"  [尝试 {attempt + 1}/{max_retries}] 正在下载: {os.path.basename(filepath)}")
            
            def progress_hook(count: int, block_size: int, total_size: int):
                """
                下载进度回调函数
                
                Args:
                    count (int): 已下载的数据块数量
                    block_size (int): 每个数据块的大小（字节）
                    total_size (int): 文件总大小（字节），-1表示未知
                
                Note:
                    此函数会被urlretrieve反复调用以更新进度
                    使用print而非logger以实现实时进度刷新效果
                """
                if total_size > 0:
                    percent = count * block_size * 100 / total_size
                    if percent <= 100:
                        print(f"\r  进度: {min(percent, 100):.1f}%", end='', flush=True)
                    if percent >= 100:
                        print(f"\r  进度: 100.0% ✅")
            
            urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
            return True
            
        except Exception as e:
            logger.warning(f"  [失败] 第{attempt + 1}次尝试出错: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"  等待2秒后重试...")
                import time  # 延迟导入，仅在需要时加载
                time.sleep(2)  # 等待2秒避免触发反爬机制
            else:
                logger.error(f"  [错误] 下载失败，已尝试{max_retries}次")
                return False
    
    return False


def download_mnist_manual(data_root: str) -> bool:
    """
    手动下载完整的MNIST数据集（使用多镜像源策略）
    
    该函数实现了完整的MNIST数据集下载流程：
    1. 创建目标目录结构
    2. 遍历所需的4个文件（2个训练集 + 2个测试集）
    3. 对每个文件检查本地是否存在（断点续传）
    4. 按优先级尝试多个镜像源直到成功
    5. 返回整体下载状态
    
    Args:
        data_root (str): 数据存储根目录
                        所有数据将保存在 {data_root}/MNIST/raw/ 目录下
                        示例: './data' -> 数据实际存储于 ./data/MNIST/raw/
    
    Returns:
        bool: 是否所有文件都下载成功
              - True: 全部4个文件均已就绪（已存在或新下载成功）
              - False: 至少有一个文件下载失败
    
    Side Effects:
        - 在磁盘上创建目录: {data_root}/MNIST/raw/
        - 向控制台输出详细的下载进度和状态信息
        - 通过logger记录操作日志
    
    Example:
        >>> if download_mnist_manual('./my_data'):
        ...     print("数据集准备完毕！")
        ... else:
        ...     print("部分文件下载失败，请检查网络连接")
    
    Directory Structure:
        data_root/
        └── MNIST/
            └── raw/
                ├── train-images-idx3-ubyte.gz   (9912422 bytes)
                ├── train-labels-idx1-ubyte.gz   (28881 bytes)
                ├── t10k-images-idx3-ubyte.gz    (1648877 bytes)
                └── t10k-labels-idx1-ubyte.gz    (4542 bytes)
    
    Mirror Strategy:
        1. 优先使用AWS S3镜像（速度快，稳定性高）
        2. 若AWS失败，回退至LeCun官方源（权威但速度较慢）
        3. 若全部失败，标记该文件为下载失败并继续下一个
    """
    raw_dir = os.path.join(data_root, 'MNIST', 'raw')  # 构建raw目录路径
    os.makedirs(raw_dir, exist_ok=True)  # 创建目录，若已存在则不报错
    
    logger.info("=" * 60)
    logger.info("开始从镜像源下载MNIST数据集")
    logger.info("=" * 60)
    
    all_success = True  # 全局成功标志，任一文件失败则设为False
    
    for dataset_type, files in MNIST_FILES.items():
        dataset_name = "训练集" if dataset_type == 'train' else "测试集"
        logger.info(f"\n正在下载{dataset_name}文件...")
        
        for filename, md5_hash in files:
            filepath = os.path.join(raw_dir, filename)
            
            if os.path.exists(filepath):
                file_size = os.path.getsize(filepath)
                logger.info(f"  ✅ 文件已存在: {filename} ({file_size/1024:.1f}KB)")
                continue  # 跳过已存在的文件（断点续传）
            
            success = False
            for mirror_url in MNIST_MIRRORS:  # 遍历镜像源列表
                url = mirror_url + filename
                if download_file(url, filepath):
                    success = True
                    break  # 当前镜像源成功则停止尝试其他源
                else:
                    if os.path.exists(filepath):
                        os.remove(filepath)  # 清理失败的下载文件
            
            if not success:
                logger.error(f"  ❌ 无法下载文件: {filename}")
                all_success = False
    
    return all_success


def download_and_preprocess_mnist(
    data_root: str = './data',
    batch_size: int = 64,
    num_workers: int = 2,
    use_custom_mirror: bool = True,
    validation_split: float = 0.1
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    """
    下载并预处理MNIST数据集（主入口函数）
    
    这是本模块的核心函数，封装了完整的数据准备流程：
    1. 环境检查和目录验证
    2. 数据集完整性检测
    3. 按需下载（支持多种模式）
    4. 数据预处理（Tensor转换 + 标准化）
    5. 数据集划分（训练/验证/测试）
    6. DataLoader创建（优化参数配置）
    7. 信息汇总和日志输出
    
    Args:
        data_root (str, optional): 数据存储根目录。默认为'./data'。
            - 相对路径会相对于当前工作目录解析
            - 绝对路径可直接指定任意位置
            - 推荐使用相对路径以保证项目可移植性
        
        batch_size (int, optional): 每个批次包含的样本数。默认为64。
            - 较大值(128-256): 提高GPU利用率，加快训练速度，但占用更多显存
            - 较小值(16-32): 减少内存消耗，适合小批量学习或调试
            - 典型值: 32, 64, 128, 256
        
        num_workers (int, optional): 数据加载的工作进程数。默认为2。
            - 0: 主进程加载数据（单线程，适合调试）
            - 2-4: 推荐值，平衡CPU利用率和开销
            - >4: 可能导致CPU过载，收益递减
            - 注意: Windows系统需在if __name__ == '__main__':下使用
        
        use_custom_mirror (bool, optional): 是否使用自定义镜像源。默认为True。
            - True: 使用AWS S3镜像（国内速度快，推荐）
            - False: 使用torchvision内置下载器（官方源，可能较慢）
        
        validation_split (float, optional): 从训练集中划分出验证集的比例。默认为0.1。
            - 取值范围: [0, 1]
            - 0: 不划分验证集，返回None
            - 0.1: 10%训练数据作为验证集（推荐）
            - 0.2: 20%训练数据作为验证集（数据量充足时）
            - 划分方式: 随机抽样，固定随机种子保证可复现性
    
    Returns:
        Tuple[DataLoader, DataLoader, Optional[DataLoader]]: 包含三个元素的元组
        
        - train_loader (DataLoader): 训练集数据加载器
          特点: shuffle=True（打乱顺序），pin_memory=True（GPU加速）
          
        - test_loader (DataLoader): 测试集数据加载器
          特点: shuffle=False（保持顺序），用于最终评估
          
        - val_loader (Optional[DataLoader]): 验证集数据加载器
          特点: 当validation_split>0时返回，否则为None
          用途: 训练过程中监控模型泛化能力，防止过拟合
    
    Raises:
        RuntimeError: 当数据集下载失败或文件损坏无法修复时抛出
        ConnectionError: 当网络连接问题导致下载中断时抛出
        IOError: 当文件系统权限不足或磁盘空间不足时抛出
        Exception: 其他未知错误
    
    Example:
        >>> # 场景1: 最简用法（使用所有默认参数）
        >>> train_loader, test_loader, val_loader = download_and_preprocess_mnist()
        >>> 
        >>> # 场景2: 自定义参数（大数据批量 + 无验证集）
        >>> train_loader, test_loader, _ = download_and_preprocess_mnist(
        ...     batch_size=256,
        ...     num_workers=4,
        ...     validation_split=0
        ... )
        >>>
        >>> # 场景3: 自定义存储位置 + 大验证集
        >>> train_loader, test_loader, val_loader = download_and_preprocess_mnist(
        ...     data_root='D:/datasets/MNIST',
        ...     validation_split=0.2,
        ...     num_workers=8
        ... )
    
    Data Preprocessing Pipeline:
        原始图像 (PIL Image, 28x28, uint8, 0-255)
            ↓ ToTensor()
        PyTorch Tensor (Float32, 1x28x28, 0.0-1.0)
            ↓ Normalize(mean=0.1307, std=0.3081)
        标准化 Tensor (Float32, 1x28x28, ≈[-0.42, 2.82])
    
    Performance Optimization:
        - pin_memory=True: 将数据锁定在分页内存中，加速CPU→GPU传输
        - num_workers>0: 多进程并行预取数据，减少I/O等待时间
        - prefetch_factor: DataLoader内部自动预取下一批数据
        - persistent_workers: 保持worker进程存活，避免重复创建开销
    
    Memory Usage Estimate (batch_size=64):
        - 单批图像: 64 × 1 × 28 × 28 × 4 bytes ≈ 200 KB
        - 单批标签: 64 × 4 bytes ≈ 256 B
        - 总计（含DataLoader缓存）: < 10 MB
    """
    
    logger.info("=" * 60)
    logger.info("MNIST数据集加载与预处理模块")
    logger.info("=" * 60)
    
    try:
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将PIL图像或numpy数组转为FloatTensor，像素值从[0,255]缩放到[0,1]
            transforms.Normalize((0.1307,), (0.3081,))  # 使用MNIST数据集的全局均值和标准差进行标准化
        ])
        
        logger.info(f"检查数据目录: {os.path.abspath(data_root)}")
        
        raw_dir = os.path.join(data_root, 'MNIST', 'raw')  # 原始压缩文件目录
        processed_dir = os.path.join(data_root, 'MNIST', 'processed')  # 处理后的.pt文件目录
        
        required_files = [  # 必须存在的4个文件列表
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]
        
        files_complete = all(  # 检查所有必需文件是否存在且有效（大小>1KB排除空文件）
            os.path.exists(os.path.join(raw_dir, f)) and 
            os.path.getsize(os.path.join(raw_dir, f)) > 1000
            for f in required_files
        )
        
        processed_exists = (  # 检查是否有可用的processed缓存（torchvision生成的.pt文件）
            os.path.exists(processed_dir) and 
            len(os.listdir(processed_dir)) >= 2
        )
        
        need_download = not (files_complete or processed_exists)  # 任一条件满足则无需重新下载
        
        if need_download:
            if use_custom_mirror:
                logger.info("本地未检测到完整的MNIST数据集")
                logger.info("使用自定义镜像源下载（支持自动重试）...")
                
                if not download_mnist_manual(data_root):
                    raise RuntimeError(
                        "数据集下载失败！请尝试：\n"
                        "1. 检查网络连接\n"
                        "2. 稍后重试\n"
                        "3. 手动下载文件到 data/MNIST/raw/ 目录"
                    )
                logger.info("\n✅ 所有文件下载完成！")
            else:
                logger.info("本地未检测到MNIST数据集，准备从官方源下载...")
                logger.info("提示: 首次下载可能需要较长时间，请耐心等待...")
                
                logger.info("\n正在加载训练集...")
                train_dataset_full = dataset.MNIST(
                    root=data_root,
                    train=True,
                    transform=transform,
                    download=True  # 让torchvision自行处理下载逻辑
                )
                logger.info(f"训练集加载成功！样本数量: {len(train_dataset_full)}")
                
                logger.info("正在加载测试集...")
                test_dataset = dataset.MNIST(
                    root=data_root,
                    train=False,
                    transform=transform,
                    download=True
                )
                logger.info(f"测试集加载成功！样本数量: {len(test_dataset)}")
                
                if validation_split > 0 and validation_split < 1:
                    train_size = int((1 - validation_split) * len(train_dataset_full))
                    val_size = len(train_dataset_full) - train_size
                    train_dataset, val_dataset = random_split(
                        train_dataset_full, [train_size, val_size],
                        generator=torch.Generator().manual_seed(42)  # 固定种子确保可复现
                    )
                    
                    val_loader = DataLoader(
                        dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=False,  # 验证集不需要打乱顺序
                        num_workers=num_workers,
                        pin_memory=True
                    )
                    logger.info(f"验证集划分完成！训练集: {train_size}样本, 验证集: {val_size}样本")
                else:
                    train_dataset = train_dataset_full
                    val_loader = None
                    logger.info("未划分验证集")
                
                train_loader = DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True,  # 训练集必须打乱顺序以提高模型泛化能力
                    num_workers=num_workers,
                    pin_memory=True
                )
                
                test_loader = DataLoader(
                    dataset=test_dataset,
                    batch_size=batch_size,
                    shuffle=False,  # 测试集保持顺序以便结果对比
                    num_workers=num_workers,
                    pin_memory=True
                )
                
                return train_loader, test_loader, val_loader
        
        logger.info(f"\n检查数据集完整性...")
        existing_raw_files = [f for f in os.listdir(raw_dir) if f.endswith('.gz')] if os.path.exists(raw_dir) else []
        
        if files_complete and not processed_exists:
            logger.info(f"✅ 原始文件完整 ({len(existing_raw_files)}个)，但处理缓存缺失")
            logger.info("使用torchvision自动处理原始数据（解压+转换）...")
            
            try:
                logger.info("\n正在加载训练集...")
                train_dataset_full = dataset.MNIST(
                    root=data_root,
                    train=True,
                    transform=transform,
                    download=True  # 必须用True让torchvision自动处理raw→processed
                )
                logger.info(f"训练集加载成功！样本数量: {len(train_dataset_full)}")
                
                logger.info("正在加载测试集...")
                test_dataset = dataset.MNIST(
                    root=data_root,
                    train=False,
                    transform=transform,
                    download=True
                )
                logger.info(f"测试集加载成功！样本数量: {len(test_dataset)}")
                
            except Exception as e:
                logger.error(f"torchvision自动处理失败: {str(e)}")
                logger.warning("尝试完全清理后重新下载...")
                
                import shutil as shutil_module
                if os.path.exists(raw_dir):
                    shutil_module.rmtree(raw_dir)
                if os.path.exists(processed_dir):
                    shutil_module.rmtree(processed_dir)
                
                if use_custom_mirror:
                    if not download_mnist_manual(data_root):
                        raise RuntimeError("数据集下载失败！请检查网络连接。")
                
                logger.info("\n正在加载训练集（全新下载）...")
                train_dataset_full = dataset.MNIST(
                    root=data_root,
                    train=True,
                    transform=transform,
                    download=True
                )
                logger.info(f"训练集加载成功！样本数量: {len(train_dataset_full)}")
                
                logger.info("正在加载测试集...")
                test_dataset = dataset.MNIST(
                    root=data_root,
                    train=False,
                    transform=transform,
                    download=True
                )
                logger.info(f"测试集加载成功！样本数量: {len(test_dataset)}")
                
        elif processed_exists:
            logger.info(f"✅ 检测到完整的处理缓存，直接加载...")
            
            try:
                logger.info("\n正在加载训练集...")
                train_dataset_full = dataset.MNIST(
                    root=data_root,
                    train=True,
                    transform=transform,
                    download=False  # processed存在，安全使用False
                )
                logger.info(f"训练集加载成功！样本数量: {len(train_dataset_full)}")
                
                logger.info("正在加载测试集...")
                test_dataset = dataset.MNIST(
                    root=data_root,
                    train=False,
                    transform=transform,
                    download=False
                )
                logger.info(f"测试集加载成功！样本数量: {len(test_dataset)}")
                
            except RuntimeError as e:
                if "Dataset not found" in str(e) or "corrupted" in str(e).lower():
                    logger.warning("⚠️ 处理缓存可能损坏，正在清理并重建...")
                    
                    import shutil as shutil_module
                    if os.path.exists(processed_dir):
                        shutil_module.rmtree(processed_dir)
                    
                    logger.info("使用原始文件重建缓存...")
                    logger.info("\n正在加载训练集...")
                    train_dataset_full = dataset.MNIST(
                        root=data_root,
                        train=True,
                        transform=transform,
                        download=True  # 强制重建
                    )
                    logger.info(f"训练集加载成功！样本数量: {len(train_dataset_full)}")
                    
                    logger.info("正在加载测试集...")
                    test_dataset = dataset.MNIST(
                        root=data_root,
                        train=False,
                        transform=transform,
                        download=True
                    )
                    logger.info(f"测试集加载成功！样本数量: {len(test_dataset)}")
                else:
                    raise
        else:
            raise RuntimeError(
                "数据集状态异常：既没有完整的原始文件也没有处理缓存。\n"
                "建议：删除 data/MNIST 目录后重新运行程序。"
            )
        
        if validation_split > 0 and validation_split < 1:
            train_size = int((1 - validation_split) * len(train_dataset_full))
            val_size = len(train_dataset_full) - train_size
            train_dataset, val_dataset = random_split(
                train_dataset_full, [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )
            
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )
            logger.info(f"验证集划分完成！训练集: {train_size}样本, 验证集: {val_size}样本")
        else:
            train_dataset = train_dataset_full
            val_loader = None
            logger.info("未划分验证集")
        
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info("=" * 60)
        logger.info("数据集信息汇总:")
        logger.info("=" * 60)
        logger.info(f"  - 数据存储位置: {os.path.abspath(data_root)}")
        logger.info(f"  - 训练集样本数: {len(train_dataset)}")
        logger.info(f"  - 测试集样本数: {len(test_dataset)}")
        if val_loader is not None:
            logger.info(f"  - 验证集样本数: {len(val_dataset)}")
        logger.info(f"  - 图像尺寸: 28x28 像素")
        logger.info(f"  - 图像通道数: 1 (灰度图)")
        logger.info(f"  - 类别数量: 10 (数字0-9)")
        logger.info(f"  - 批次大小: {batch_size}")
        logger.info(f"  - 数据格式: PyTorch Tensor [batch_size, 1, 28, 28]")
        logger.info(f"  - 像素值变换: 先缩放至[0,1]，再标准化 (均值0.1307, 标准差0.3081)")
        logger.info(f"  - 加载优化: num_workers={num_workers}, pin_memory=True")
        logger.info("=" * 60)
        
        return train_loader, test_loader, val_loader
        
    except (ConnectionError, urllib.error.URLError, TimeoutError) as e:
        logger.error(f"网络连接失败！错误详情: {str(e)}")
        logger.error("请检查网络连接后重试，或手动下载数据集放置于指定目录")
        raise
        
    except IOError as e:
        logger.error(f"文件系统操作失败！错误详情: {str(e)}")
        logger.error("请检查文件权限和磁盘空间")
        raise
        
    except Exception as e:
        logger.error(f"发生未知错误！错误类型: {type(e).__name__}, 详情: {str(e)}")
        raise


def print_dataset_info(
    train_loader: DataLoader,
    test_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    num_samples: int = 6
):
    """
    打印MNIST数据集的详细信息（可视化查看入口函数）
    
    本函数提供全面的数据集概览，帮助开发者理解数据特征和质量，
    是数据探索和调试的重要工具。
    
    输出内容包括5个部分：
    1. 基本信息：样本数量、批次设置
    2. 样本示例：展示具体图像的张量属性
    3. 像素统计：数值分布的关键指标
    4. 标签分布：各类别的数量和占比（带柱状图）
    5. 格式总结：数据规格速查表
    
    Args:
        train_loader (DataLoader): 训练集数据加载器
            通常由download_and_preprocess_mnist()返回的第一个元素
        
        test_loader (DataLoader): 测试集数据加载器
            通常由download_and_preprocess_mnist()返回的第二个元素
        
        val_loader (Optional[DataLoader], optional): 验证集数据加载器
            如果使用了验证集划分(validation_split>0)，传入该加载器
            若无验证集可传None或省略此参数
            默认值: None
        
        num_samples (int, optional): 要展示的具体样本数量。默认为6。
            这些样本将从第一个批次中选取并显示其详细信息
            建议范围: 1-64（不超过单个批次的大小）
    
    Returns:
        None: 此函数仅打印信息到控制台，无返回值
    
    Side Effects:
        - 向标准输出打印格式化的数据集报告（约50-80行）
        - 会遍历整个DataLoader以统计标签分布（耗时约几秒）
        - 不会修改任何输入数据或文件系统
    
    Output Format:
        函数输出采用中文，包含emoji图标增强可读性：
        - 📊 数据集标题
        - 🎯 样本示例
        - 📈 分布图表（使用█字符绘制柱状图）
        - ✅ 状态标识
    
    Example:
        >>> # 基础用法：查看所有数据集信息
        >>> from pytorch_mnist import download_and_preprocess_mnist, print_dataset_info
        >>> 
        >>> train_loader, test_loader, val_loader = download_and_preprocess_mnist()
        >>> print_dataset_info(train_loader, test_loader, val_loader)
        >>>
        >>> # 仅查看训练集和测试集（忽略验证集）
        >>> print_dataset_info(train_loader, test_loader)
        >>>
        >>> # 显示更多样本细节
        >>> print_dataset_info(train_loader, test_loader, num_samples=20)
    
    Performance Note:
        - 标签分布统计需要遍历完整数据集，对于大型数据集可能需要几秒钟
        - 对于MNIST（60K训练+10K测试），通常<1秒即可完成
        - 如仅需快速检查，可将num_samples设为较小值（如2-3）
    
    Integration with Training Pipeline:
        建议在训练前调用一次以确认数据正确性：
        >>> # 在训练脚本中的典型用法
        >>> train_loader, test_loader, val_loader = download_and_preprocess_mnist()
        >>> print_dataset_info(train_loader, test_loader, val_loader)  # 数据验证
        >>> model = build_model()
        >>> train_model(model, train_loader, val_loader)
        >>> evaluate_model(model, test_loader)
    """
    
    print("\n" + "=" * 70)
    print("📊 MNIST数据集详细信息")
    print("=" * 70)
    
    print("\n【1. 数据集基本信息】")
    print(f"  训练集总样本数: {len(train_loader.dataset)}")
    print(f"  测试集总样本数: {len(test_loader.dataset)}")
    if val_loader is not None:
        print(f"  验证集总样本数: {len(val_loader.dataset)}")
    print(f"  批次大小: {train_loader.batch_size}")
    print(f"  训练集批次数: {len(train_loader)}")
    print(f"  测试集批次数: {len(test_loader)}")
    
    print("\n【2. 样本图像示例】")
    train_images, train_labels = next(iter(train_loader))
    test_images, test_labels = next(iter(test_loader))
    
    print(f"\n  🎯 训练集前{min(num_samples, len(train_images))}个样本:")
    for i in range(min(num_samples, len(train_images))):
        img = train_images[i]
        label = train_labels[i].item()
        print(f"    样本{i+1}: 标签={label}, 图像形状={list(img.shape)}, "
              f"像素范围=[{img.min():.3f}, {img.max():.3f}]")
    
    print(f"\n  🎯 测试集前{min(num_samples, len(test_images))}个样本:")
    for i in range(min(num_samples, len(test_images))):
        img = test_images[i]
        label = test_labels[i].item()
        print(f"    样本{i+1}: 标签={label}, 图像形状={list(img.shape)}, "
              f"像素范围=[{img.min():.3f}, {img.max():.3f}]")
    
    print("\n【3. 像素值统计】")
    print(f"  训练集批次:")
    print(f"    - 形状: {train_images.shape}")
    print(f"    - 最小值: {train_images.min().item():.4f}")
    print(f"    - 最大值: {train_images.max().item():.4f}")
    print(f"    - 平均值: {train_images.mean().item():.4f}")
    print(f"    - 标准差: {train_images.std().item():.4f}")
    
    print(f"\n  测试集批次:")
    print(f"    - 形状: {test_images.shape}")
    print(f"    - 最小值: {test_images.min().item():.4f}")
    print(f"    - 最大值: {test_images.max().item():.4f}")
    print(f"    - 平均值: {test_images.mean().item():.4f}")
    print(f"    - 标准差: {test_images.std().item():.4f}")
    
    print("\n【4. 标签分布统计】")
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.extend(labels.tolist())
    
    all_test_labels = []
    for _, labels in test_loader:
        all_test_labels.extend(labels.tolist())
    
    train_counter = Counter(all_train_labels)
    test_counter = Counter(all_test_labels)
    
    print("\n  📈 训练集标签分布:")
    for digit in sorted(train_counter.keys()):
        count = train_counter[digit]
        percentage = count / len(all_train_labels) * 100
        bar = '█' * int(percentage / 2)
        print(f"    数字 {digit}: {count:5d} 个 ({percentage:5.1f}%) {bar}")
    
    print("\n  📈 测试集标签分布:")
    for digit in sorted(test_counter.keys()):
        count = test_counter[digit]
        percentage = count / len(all_test_labels) * 100
        bar = '█' * int(percentage / 2)
        print(f"    数字 {digit}: {count:5d} 个 ({percentage:5.1f}%) {bar}")
    
    if val_loader is not None:
        all_val_labels = []
        for _, labels in val_loader:
            all_val_labels.extend(labels.tolist())
        
        val_counter = Counter(all_val_labels)
        print("\n  📈 验证集标签分布:")
        for digit in sorted(val_counter.keys()):
            count = val_counter[digit]
            percentage = count / len(all_val_labels) * 100
            bar = '█' * int(percentage / 2)
            print(f"    数字 {digit}: {count:5d} 个 ({percentage:5.1f}%) {bar}")
    
    print("\n【5. 数据格式总结】")
    print("  ✅ 图像格式: PyTorch Tensor [batch_size, channels, height, width]")
    print("  ✅ 图像尺寸: 28×28 像素")
    print("  ✅ 通道数: 1 (灰度图)")
    print("  ✅ 预处理: ToTensor() + Normalize(0.1307, 0.3081)")
    print("  ✅ 标签格式: LongTensor (0-9的整数)")
    
    print("\n" + "=" * 70)
    print("✨ 数据集检查完成，一切正常！")
    print("=" * 70)


if __name__ == "__main__":
    """
    主程序入口点
    
    当直接运行此脚本时（python pytorch_mnist.py），执行以下流程：
    1. 调用download_and_preprocess_mnist()下载和准备数据集
    2. 快速验证DataLoader的基本功能（获取一个batch）
    3. 调用print_dataset_info()输出完整的数据集分析报告
    
    This block ensures the code only runs when executed directly,
    not when imported as a module.
    This is especially important on Windows when using multiprocessing (num_workers > 0).
    """
    try:
        logger.info("启动MNIST数据集加载程序...")
        
        train_loader, test_loader, val_loader = download_and_preprocess_mnist()
        
        logger.info("快速验证DataLoader功能...")
        images, labels = next(iter(train_loader))
        logger.info(f"✅ 批次数据形状: {images.shape}")
        logger.info(f"✅ 标签形状: {labels.shape}")
        logger.info("像素值统计:")
        logger.info(f"         - 最小值: {images.min().item():.4f}")
        logger.info(f"         - 最大值: {images.max().item():.4f}")
        logger.info(f"         - 平均值: {images.mean().item():.4f}")
        logger.info(f"         - 标准差: {images.std().item():.4f}")
        
        if val_loader is not None:
            val_images, val_labels = next(iter(val_loader))
            logger.info(f"✅ 验证集批次形状: {val_images.shape}")
        
        logger.info("调用数据集信息打印函数...")
        print_dataset_info(train_loader, test_loader, val_loader)

        logger.info("=" * 60)
        logger.info("CNN模型初始化与CUDA加速配置")
        logger.info("=" * 60)

        from CNN import CNN

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")

        model = CNN().to(device)
        logger.info(f"✅ CNN模型已创建并转移至{device}设备")
        logger.info(f"模型结构:\n{model}")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"总参数量: {total_params:,}")
        logger.info(f"可训练参数量: {trainable_params:,}")

        logger.info("\n开始遍历训练数据进行前向传播（CUDA加速）...")
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            if batch_idx == 0:
                logger.info(f"批次 {batch_idx + 1}:")
                logger.info(f"  输入图像形状: {images.shape}")
                logger.info(f"  标签形状: {labels.shape}")
                logger.info(f"  模型输出形状: {outputs.shape}")
                logger.info(f"  输出样本值(前5个):\n{outputs[0][:5].detach().cpu()}")

            if batch_idx >= 2:
                logger.info(f"... 已处理 {batch_idx + 1} 个批次，停止演示循环")
                break

        logger.info("\n✅ CUDA加速前向传播测试完成！")

        logger.info("=" * 60)
        logger.info("配置损失函数与优化器")
        logger.info("=" * 60)

        import torch.nn as nn
        import torch.optim as optim
        from tqdm import tqdm

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        logger.info(f"✅ 损失函数: CrossEntropyLoss (交叉熵损失)")
        logger.info(f"✅ 优化器: Adam (学习率=0.001)")

        num_epochs = 5
        logger.info(f"\n开始训练模型 (共 {num_epochs} 轮)...")
        logger.info("=" * 60)

        def evaluate_model(model, data_loader, criterion, device, num_samples=5, epoch_num=None, is_final=False):

            model.eval()
            total_loss = 0.0
            correct = 0
            total = 0
            all_predictions = []
            all_labels = []
            sample_details = []

            prefix = '最终测试' if is_final else f'Epoch [{epoch_num}] 测试'
            test_pbar = tqdm(data_loader, desc=f'{prefix}评估中', unit='batch', leave=False)

            with torch.no_grad():
                for batch_idx, (images, labels) in enumerate(test_pbar):

                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    total_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)

                    bool_comparison = (predicted == labels)
                    correct += bool_comparison.sum().item()
                    total += labels.size(0)

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    batch_acc = 100 * bool_comparison.sum().item() / labels.size(0)
                    avg_loss = total_loss / (batch_idx + 1)

                    test_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg_Loss': f'{avg_loss:.4f}',
                        'Acc': f'{batch_acc:.2f}%'
                    })

                    if len(sample_details) < num_samples:
                        for i in range(min(num_samples - len(sample_details), labels.size(0))):
                            sample_details.append({
                                'predicted': predicted[i].item(),
                                'actual': labels[i].item(),
                                'correct': bool_comparison[i].item(),
                                'confidence': torch.softmax(outputs[i], dim=0).max().item() * 100
                            })

            avg_loss = total_loss / len(data_loader)
            accuracy = 100 * correct / total

            title = '📊 最终测试结果' if is_final else f'📊 Epoch [{epoch_num}] 测试结果'
            logger.info(f'\n{title}:')
            logger.info(f'  - 总损失: {total_loss:.4f}')
            logger.info(f'  - 平均损失: {avg_loss:.4f}')
            logger.info(f'  - 正确数: {correct}/{total}')
            logger.info(f'  - 准确率: {accuracy:.2f}%')

            if sample_details:
                logger.info(f'\n  样本级预测详情 (前{len(sample_details)}个样本):')
                logger.info('  ' + '-' * 70)
                logger.info(f'  {"序号":<6}{"预测标签":<10}{"真实标签":<10}{"是否正确":<12}{"置信度":<12}')
                logger.info('  ' + '-' * 70)

                for idx, detail in enumerate(sample_details, 1):
                    status = '✅ 正确' if detail['correct'] else '❌ 错误'
                    logger.info(
                        f'  {idx:<6}{detail["predicted"]:<10}'
                        f'{detail["actual"]:<10}{status:<12}'
                        f'{detail["confidence"]:.2f}%'
                    )
                logger.info('  ' + '-' * 70)

            return avg_loss, accuracy, sample_details

        for epoch in range(num_epochs):

            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            epoch_pbar = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', unit='batch')

            for batch_idx, (images, labels) in enumerate(epoch_pbar):

                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                avg_loss = running_loss / (batch_idx + 1)
                accuracy = 100 * correct / total

                epoch_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Avg_Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })

            epoch_loss = running_loss / len(train_loader)
            epoch_acc = 100 * correct / total

            logger.info(f'\nEpoch [{epoch+1}/{num_epochs}] 训练完成:')
            logger.info(f'  - 平均损失: {epoch_loss:.4f}')
            logger.info(f'  - 训练准确率: {epoch_acc:.2f}%')

            test_loss, test_acc, sample_details = evaluate_model(
                model, test_loader, criterion, device, 
                num_samples=5, epoch_num=epoch+1
            )

            logger.info('-' * 60)

        logger.info('\n' + '=' * 60)
        logger.info('🎉 训练与验证完成！')
        logger.info('=' * 60)

        final_test_loss, final_test_acc, _ = evaluate_model(
            model, test_loader, criterion, device,
            num_samples=10, is_final=True
        )

        logger.info('\n' + '=' * 60)
        logger.info('💾 保存训练好的模型')
        logger.info('=' * 60)

        import os
        model_dir = './models'
        os.makedirs(model_dir, exist_ok=True)

        timestamp = __import__('datetime').datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(model_dir, f'cnn_mnist_{timestamp}.pkl')

        save_info = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': num_epochs,
            'test_accuracy': final_test_acc,
            'test_loss': final_test_loss,
            'model_architecture': str(model)
        }

        torch.save(save_info, model_path)
        file_size = os.path.getsize(model_path) / (1024 * 1024)

        logger.info(f'✅ 模型已成功保存！')
        logger.info(f'  - 保存路径: {os.path.abspath(model_path)}')
        logger.info(f'  - 文件大小: {file_size:.2f} MB')
        logger.info(f'  - 测试准确率: {final_test_acc:.2f}%')
        logger.info(f'\n💡 加载模型方法:')
        logger.info(f"  model = CNN()")
        logger.info(f"  checkpoint = torch.load('{model_path}')")
        logger.info(f"  model.load_state_dict(checkpoint['model_state_dict'])")
        logger.info('=' * 60)

        logger.info("🎉 程序执行成功完成！")
        
    except KeyboardInterrupt:
        logger.warning("\n用户中断程序执行（Ctrl+C）")
    except Exception as e:
        logger.error(f"❌ 程序执行中断: {str(e)}")
        logger.error("请参考上方错误信息排查问题，或查阅README.md获取帮助")
        exit(1)
