"""
图文检索系统 - 使用CLIP和FAISS
支持文本检索图片和图片检索文本
适用于Google Colab和本地运行
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import faiss
import clip
from typing import List, Tuple, Optional
import json
import pickle


class ImageTextRetrieval:
    def __init__(
        self,
        device: str = None,
        clip_model: str = "ViT-B/32",
        index_type: str = "flat"
    ):
        """
        初始化图文检索系统
        
        Args:
            device: 计算设备 (cuda/cpu/mps)
            clip_model: CLIP模型名称
            index_type: FAISS索引类型 (flat/ivf/hnsw)
        """
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        self.clip_model_name = clip_model
        self.index_type = index_type
        
        self.model = None
        self.preprocess = None
        self.image_index = None
        self.text_index = None
        self.image_embeddings = None
        self.text_embeddings = None
        self.image_paths = []
        self.texts = []
        self.metadata = {}
        
        self._load_clip_model()
    
    def _load_clip_model(self):
        """加载CLIP模型"""
        print(f"加载CLIP模型: {self.clip_model_name}")
        self.model, self.preprocess = clip.load(
            self.clip_model_name, 
            device=self.device
        )
        self.model.eval()
        print("CLIP模型加载完成")
    
    def encode_images(self, image_paths: List[str], batch_size: int = 32) -> np.ndarray:
        """
        编码图片为向量
        
        Args:
            image_paths: 图片路径列表
            batch_size: 批处理大小
            
        Returns:
            图片向量数组 (N, 512)
        """
        print(f"编码 {len(image_paths)} 张图片...")
        embeddings = []
        
        for i in tqdm(range(0, len(image_paths), batch_size)):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    image = self.preprocess(image)
                    batch_images.append(image)
                except Exception as e:
                    print(f"加载图片失败 {path}: {e}")
                    batch_images.append(torch.zeros(3, 224, 224))
            
            batch_images = torch.stack(batch_images).to(self.device)
            
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(batch_images)
                batch_embeddings = F.normalize(batch_embeddings, dim=-1)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        self.image_embeddings = np.vstack(embeddings).astype('float32')
        self.image_paths = image_paths
        print(f"图片编码完成，形状: {self.image_embeddings.shape}")
        
        return self.image_embeddings
    
    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        编码文本为向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            文本向量数组 (N, 512)
        """
        print(f"编码 {len(texts)} 条文本...")
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i + batch_size]
            
            with torch.no_grad():
                text_tokens = clip.tokenize(
                    batch_texts, 
                    truncate=True
                ).to(self.device)
                batch_embeddings = self.model.encode_text(text_tokens)
                batch_embeddings = F.normalize(batch_embeddings, dim=-1)
            
            embeddings.append(batch_embeddings.cpu().numpy())
        
        self.text_embeddings = np.vstack(embeddings).astype('float32')
        self.texts = texts
        print(f"文本编码完成，形状: {self.text_embeddings.shape}")
        
        return self.text_embeddings
    
    def build_index(
        self, 
        embeddings: np.ndarray, 
        nlist: int = 100,
        nprobe: int = 10
    ) -> faiss.Index:
        """
        构建FAISS索引
        
        Args:
            embeddings: 向量数组
            nlist: IVF聚类中心数量
            nprobe: 搜索时探测的聚类数量
            
        Returns:
            FAISS索引
        """
        d = embeddings.shape[1]
        
        if self.index_type == "flat":
            index = faiss.IndexFlatIP(d)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist)
            index.train(embeddings)
            index.nprobe = nprobe
        elif self.index_type == "hnsw":
            M = 32
            index = faiss.IndexHNSWFlat(d, M)
            index.hnsw.efSearch = 64
        else:
            raise ValueError(f"不支持的索引类型: {self.index_type}")
        
        index.add(embeddings)
        print(f"索引构建完成，包含 {index.ntotal} 个向量")
        
        return index
    
    def build_image_index(self, nlist: int = 100):
        """构建图片向量索引"""
        if self.image_embeddings is None:
            raise ValueError("请先编码图片")
        self.image_index = self.build_index(self.image_embeddings, nlist)
        return self.image_index
    
    def build_text_index(self, nlist: int = 100):
        """构建文本向量索引"""
        if self.text_embeddings is None:
            raise ValueError("请先编码文本")
        self.text_index = self.build_index(self.text_embeddings, nlist)
        return self.text_index
    
    def search_text_to_image(
        self, 
        query: str, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        文本检索图片
        
        Args:
            query: 查询文本
            k: 返回结果数量
            
        Returns:
            [(图片路径, 相似度分数), ...]
        """
        with torch.no_grad():
            text_token = clip.tokenize([query], truncate=True).to(self.device)
            query_embedding = self.model.encode_text(text_token)
            query_embedding = F.normalize(query_embedding, dim=-1)
            query_embedding = query_embedding.cpu().numpy().astype('float32')
        
        D, I = self.image_index.search(query_embedding, k)
        
        results = []
        for i, (idx, score) in enumerate(zip(I[0], D[0])):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(score)))
        
        return results
    
    def search_image_to_text(
        self, 
        image_path: str, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        图片检索文本
        
        Args:
            image_path: 查询图片路径
            k: 返回结果数量
            
        Returns:
            [(文本, 相似度分数), ...]
        """
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.model.encode_image(image)
            query_embedding = F.normalize(query_embedding, dim=-1)
            query_embedding = query_embedding.cpu().numpy().astype('float32')
        
        D, I = self.text_index.search(query_embedding, k)
        
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.texts):
                results.append((self.texts[idx], float(score)))
        
        return results
    
    def search_image_to_image(
        self, 
        image_path: str, 
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        图片检索图片（以图搜图）
        
        Args:
            image_path: 查询图片路径
            k: 返回结果数量
            
        Returns:
            [(图片路径, 相似度分数), ...]
        """
        image = Image.open(image_path).convert('RGB')
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            query_embedding = self.model.encode_image(image)
            query_embedding = F.normalize(query_embedding, dim=-1)
            query_embedding = query_embedding.cpu().numpy().astype('float32')
        
        D, I = self.image_index.search(query_embedding, k + 1)
        
        results = []
        for idx, score in zip(I[0], D[0]):
            if idx < len(self.image_paths) and self.image_paths[idx] != image_path:
                results.append((self.image_paths[idx], float(score)))
        
        return results[:k]
    
    def visualize_results(
        self, 
        query: str, 
        results: List[Tuple[str, float]], 
        query_type: str = "text"
    ):
        """
        可视化检索结果
        
        Args:
            query: 查询内容
            results: 检索结果
            query_type: 查询类型 (text/image)
        """
        n = len(results)
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))
        
        if query_type == "image":
            query_img = Image.open(query).convert('RGB')
            axes[0].imshow(query_img)
            axes[0].set_title("查询图片", fontsize=12)
            axes[0].axis('off')
            
            for i, (text, score) in enumerate(results):
                axes[i + 1].text(
                    0.5, 0.5, text, 
                    ha='center', va='center',
                    fontsize=10, wrap=True,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
                )
                axes[i + 1].set_title(f"相似度: {score:.4f}")
                axes[i + 1].axis('off')
        else:
            axes[0].text(
                0.5, 0.5, query, 
                ha='center', va='center',
                fontsize=12, wrap=True,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5)
            )
            axes[0].set_title("查询文本", fontsize=12)
            axes[0].axis('off')
            
            for i, (img_path, score) in enumerate(results):
                try:
                    img = Image.open(img_path).convert('RGB')
                    axes[i + 1].imshow(img)
                    axes[i + 1].set_title(f"相似度: {score:.4f}", fontsize=10)
                except Exception as e:
                    axes[i + 1].text(0.5, 0.5, "加载失败", ha='center', va='center')
                axes[i + 1].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def save_index(self, save_dir: str):
        """保存索引和数据"""
        os.makedirs(save_dir, exist_ok=True)
        
        if self.image_index is not None:
            faiss.write_index(self.image_index, os.path.join(save_dir, "image_index.faiss"))
        
        if self.text_index is not None:
            faiss.write_index(self.text_index, os.path.join(save_dir, "text_index.faiss"))
        
        metadata = {
            'image_paths': self.image_paths,
            'texts': self.texts,
            'clip_model': self.clip_model_name,
            'index_type': self.index_type
        }
        
        with open(os.path.join(save_dir, "metadata.pkl"), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"索引已保存到 {save_dir}")
    
    def load_index(self, save_dir: str):
        """加载索引和数据"""
        image_index_path = os.path.join(save_dir, "image_index.faiss")
        text_index_path = os.path.join(save_dir, "text_index.faiss")
        metadata_path = os.path.join(save_dir, "metadata.pkl")
        
        if os.path.exists(image_index_path):
            self.image_index = faiss.read_index(image_index_path)
        
        if os.path.exists(text_index_path):
            self.text_index = faiss.read_index(text_index_path)
        
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.image_paths = metadata.get('image_paths', [])
            self.texts = metadata.get('texts', [])


def download_coco_dataset(sample_size: int = 1000, save_dir: str = "./coco_data"):
    """
    下载COCO数据集样本
    
    Args:
        sample_size: 样本数量
        save_dir: 保存目录
    """
    import requests
    import zipfile
    from pycocotools.coco import COCO
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("下载COCO标注文件...")
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_zip = os.path.join(save_dir, "annotations.zip")
    
    if not os.path.exists(ann_zip):
        response = requests.get(ann_url, stream=True)
        with open(ann_zip, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
    
    ann_file = os.path.join(save_dir, "annotations", "captions_val2017.json")
    coco = COCO(ann_file)
    
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    img_ids = list(coco.imgs.keys())[:sample_size]
    
    image_paths = []
    texts = []
    
    print(f"下载 {len(img_ids)} 张图片...")
    for img_id in tqdm(img_ids):
        img_info = coco.loadImgs(img_id)[0]
        img_url = img_info['coco_url']
        img_path = os.path.join(img_dir, img_info['file_name'])
        
        if not os.path.exists(img_path):
            try:
                response = requests.get(img_url, stream=True)
                with open(img_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                print(f"下载失败 {img_url}: {e}")
                continue
        
        image_paths.append(img_path)
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        if anns:
            texts.append(anns[0]['caption'])
    
    print(f"数据集准备完成: {len(image_paths)} 张图片, {len(texts)} 条文本")
    return image_paths, texts


def create_sample_dataset(save_dir: str = "./sample_data"):
    """
    创建示例数据集（用于快速测试）
    使用随机生成的图片和文本
    """
    os.makedirs(save_dir, exist_ok=True)
    img_dir = os.path.join(save_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    
    sample_texts = [
        "a dog running in the park",
        "a cat sleeping on the sofa",
        "a bird flying in the sky",
        "a car driving on the road",
        "a boat sailing on the sea",
        "a person riding a bicycle",
        "a child playing with toys",
        "a flower blooming in the garden",
        "a mountain covered with snow",
        "a city skyline at night"
    ]
    
    image_paths = []
    
    print("创建示例图片...")
    for i, text in enumerate(tqdm(sample_texts)):
        img_path = os.path.join(img_dir, f"sample_{i}.jpg")
        
        if not os.path.exists(img_path):
            colors = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(colors)
            img.save(img_path)
        
        image_paths.append(img_path)
    
    print(f"示例数据集创建完成: {len(image_paths)} 张图片")
    return image_paths, sample_texts


def demo_with_coco():
    """使用COCO数据集演示"""
    print("=" * 60)
    print("图文检索系统演示 - COCO数据集")
    print("=" * 60)
    
    retrieval = ImageTextRetrieval(device="cuda", index_type="flat")
    
    image_paths, texts = download_coco_dataset(sample_size=500)
    
    retrieval.encode_images(image_paths)
    retrieval.encode_texts(texts)
    
    retrieval.build_image_index()
    retrieval.build_text_index()
    
    retrieval.save_index("./retrieval_index")
    
    print("\n" + "=" * 60)
    print("文本检索图片测试")
    print("=" * 60)
    
    test_queries = [
        "a dog playing in the park",
        "people walking on the street",
        "a car on the road"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = retrieval.search_text_to_image(query, k=3)
        for i, (path, score) in enumerate(results):
            print(f"  {i+1}. {os.path.basename(path)} (相似度: {score:.4f})")
        
        retrieval.visualize_results(query, results, query_type="text")
    
    print("\n" + "=" * 60)
    print("图片检索文本测试")
    print("=" * 60)
    
    test_image = image_paths[0]
    print(f"\n查询图片: {os.path.basename(test_image)}")
    results = retrieval.search_image_to_text(test_image, k=3)
    for i, (text, score) in enumerate(results):
        print(f"  {i+1}. {text} (相似度: {score:.4f})")


def demo_with_sample():
    """使用示例数据演示"""
    print("=" * 60)
    print("图文检索系统演示 - 示例数据")
    print("=" * 60)
    
    retrieval = ImageTextRetrieval(device="cpu", index_type="flat")
    
    image_paths, texts = create_sample_dataset()
    
    retrieval.encode_images(image_paths)
    retrieval.encode_texts(texts)
    
    retrieval.build_image_index()
    retrieval.build_text_index()
    
    print("\n" + "=" * 60)
    print("文本检索图片测试")
    print("=" * 60)
    
    test_queries = [
        "a cute animal",
        "outdoor scenery",
        "transportation vehicle"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        results = retrieval.search_text_to_image(query, k=3)
        for i, (path, score) in enumerate(results):
            print(f"  {i+1}. {os.path.basename(path)} (相似度: {score:.4f})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="图文检索系统")
    parser.add_argument("--mode", type=str, default="sample", choices=["sample", "coco"],
                       help="运行模式: sample(示例数据) 或 coco(COCO数据集)")
    parser.add_argument("--device", type=str, default=None,
                       help="计算设备: cuda/cpu/mps")
    parser.add_argument("--index_type", type=str, default="flat",
                       choices=["flat", "ivf", "hnsw"],
                       help="FAISS索引类型")
    
    args = parser.parse_args()
    
    if args.mode == "coco":
        demo_with_coco()
    else:
        demo_with_sample()
