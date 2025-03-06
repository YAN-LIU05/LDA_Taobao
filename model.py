"""
淘宝评论深度语义分析系统
环境要求：Python 3.8+，需提前安装以下库：
pip install pandas openpyxl jieba scikit-learn transformers torch textrank4zh matplotlib wordcloud tqdm networkx==2.5
sudo apt-get update
sudo apt-get install fonts-wqy-microhei
"""

import pandas as pd
import re
import jieba
import numpy as np
import torch
import logging
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModel
from textrank4zh import TextRank4Keyword, TextRank4Sentence
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import openpyxl
from tqdm import tqdm
from openpyxl.drawing.image import Image
from matplotlib.font_manager import FontProperties

font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 
font = FontProperties(fname=font_path)
label = 3
# ======================
# 配置参数
# ======================
DEFAULT_CONFIG = {
    "input_file": f"/root/autodl-tmp/comments_{label}.xlsx",  # 输入文件路径
    "output_file": f"/root/autodl-tmp/analysis_report_{label}.xlsx",  # 输出文件路径
    "text_column": "评论内容",  # 待分析的文本列名
    "custom_dict": "/root/autodl-tmp/custom_dict.txt",  # 自定义词典路径
    "stopwords": "/root/autodl-tmp/stopwords.txt",  # 停用词文件路径
    "max_length": 300,  # 最大处理文本长度
    "cluster_range": (3, 8),  # 聚类数量搜索范围
    "device": "cuda" if torch.cuda.is_available() else "cpu",  # 自动选择设备
    "model_name": "hfl/chinese-roberta-wwm-ext",  # 预训练模型名称
    "local_model_path": "/root/autodl-tmp/local_model",  # 本地模型路径（优先使用）
}

# ======================
# 工具函数
# ======================
def load_resources(config):
    """加载外部资源"""
    # 加载自定义词典
    if config["custom_dict"]:
        try:
            jieba.load_userdict(config["custom_dict"])
            logging.info("自定义词典加载成功")
        except Exception as e:
            logging.warning(f"自定义词典加载失败: {e}")

    # 加载停用词
    stopwords = set()
    if config["stopwords"]:
        try:
            with open(config["stopwords"], encoding="utf-8") as f:
                stopwords = set(f.read().split())
            logging.info("停用词表加载成功")
        except Exception as e:
            logging.warning(f"停用词表加载失败: {e}")

    return stopwords


def enhanced_clean(text):
    """增强型文本清洗"""
    if pd.isna(text):
        return ""

    # 去除电商特定内容
    text = re.sub(
        r'【.*?】|@\S+|(颜色|尺码|型号)：\S+|【点击领取】|[0-9]{11}|(http|https)://\S+',
        '',
        str(text)
    )
    # 保留有效字符
    text = re.sub(r'[^\u4e00-\u9fa5，。！？、；：“”‘’\s]', '', text)
    return text.strip()[:CONFIG["max_length"]]


# ======================
# 核心处理类
# ======================
class CommentAnalyzer:
    def __init__(self, config):
        self.config = config
        self.stopwords = load_resources(config)

        # 初始化模型
        self.tokenizer, self.model = self.load_model()

        # 初始化TextRank
        self.tr4w = TextRank4Keyword()
        self.tr4s = TextRank4Sentence()

    def load_model(self):
        """加载预训练模型"""
        """模型下载地址
        https://huggingface.co/hfl/chinese-roberta-wwm-ext/tree/main
        下载 config.json pytorch_model.bin vocab.txt
        """
        try:
            if self.config["local_model_path"]:
                logging.info(f"从本地加载模型: {self.config['local_model_path']}")
                tokenizer = AutoTokenizer.from_pretrained(self.config["local_model_path"])
                model = AutoModel.from_pretrained(self.config["local_model_path"]).to(self.config["device"])
            else:
                logging.info(f"从Hugging Face加载模型: {self.config['model_name']}")
                tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"], timeout=30)
                model = AutoModel.from_pretrained(self.config["model_name"], timeout=30).to(self.config["device"])
            return tokenizer, model
        except Exception as e:
            logging.error(f"模型加载失败: {e}")
            raise

    def process_file(self):
        """主处理流程"""
        try:
            # 数据加载与清洗
            df = self.load_and_clean_data()

            # 深度分析
            df = self.semantic_analysis(df)

            # 聚类分析
            df = self.cluster_analysis(df)

            # 生成报告
            self.generate_report(df)
        except Exception as e:
            logging.error(f"处理过程中发生错误: {e}")
            raise

    def load_and_clean_data(self):
        """数据加载与清洗"""
        logging.info("正在加载数据...")
        try:
            df = pd.read_excel(
                self.config["input_file"],
                engine="openpyxl",
                usecols=[self.config["text_column"]]
            )
            logging.info(f"原始数据量: {len(df)}")

            # 数据清洗
            df["cleaned"] = df[self.config["text_column"]].apply(enhanced_clean)
            df = df[df["cleaned"].str.len() > 3]  # 过滤短文本
            logging.info(f"有效数据量: {len(df)}")
            return df
        except Exception as e:
            logging.error(f"数据加载失败: {e}")
            raise

    def semantic_analysis(self, df):
        """语义分析"""
        logging.info("正在执行语义分析...")
        try:
            # 批量生成嵌入向量
            df["embedding"] = df["cleaned"].progress_apply(
                lambda x: self.get_embedding(x)
            )

            # 关键词提取
            def extract_keywords_with_logging(text):
                try:
                    return self.extract_keywords(text, 5)
                except Exception as e:
                    logging.error(f"提取关键词时出错: {e}")
                    return []  # 返回空列表以防止中断

            df["keywords"] = df["cleaned"].progress_apply(extract_keywords_with_logging)

            # 摘要生成
            df["summary"] = df["cleaned"].progress_apply(
                lambda x: self.extract_summary(x, 2)
            )
            return df
        except Exception as e:
            logging.error(f"语义分析失败: {e}")
            raise


    def get_embedding(self, text):
        """生成文本嵌入向量"""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=128,
            truncation=True,
            padding=True
        ).to(self.config["device"])

        with torch.no_grad():
            outputs = self.model(**inputs)

        return torch.mean(outputs.last_hidden_state, dim=1).squeeze().cpu().numpy()

    def extract_keywords(self, text, num=5):
        """提取关键词"""
        self.tr4w.analyze(text, window=5, lower=True)
        return [k.word for k in self.tr4w.get_keywords(num)]

    def extract_summary(self, text, num=2):
        """生成摘要"""
        self.tr4s.analyze(text, lower=True)
        return [s.sentence for s in self.tr4s.get_key_sentences(num)]

    def cluster_analysis(self, df):
        """聚类分析"""
        logging.info("正在执行聚类分析...")
        try:
            # 降维处理
            pca = PCA(n_components=0.95)
            embeddings = np.vstack(df["embedding"].values)
            reduced_embeds = pca.fit_transform(embeddings)

            # 自动确定最佳聚类数
            best_k = self.find_optimal_clusters(reduced_embeds)
            logging.info(f"自动选择聚类数: {best_k}")

            # 执行聚类
            kmeans = KMeans(n_clusters=best_k, random_state=42)
            df["cluster"] = kmeans.fit_predict(reduced_embeds)
            return df
        except Exception as e:
            logging.error(f"聚类分析失败: {e}")
            raise

    def find_optimal_clusters(self, data):
        """使用肘部法则确定最佳聚类数"""
        distortions = []
        K = range(*self.config["cluster_range"])

        for k in tqdm(K, desc="寻找最佳聚类数"):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)
            distortions.append(kmeans.inertia_)

        # 计算曲率变化
        diff = np.diff(distortions)
        return np.argmin(diff) + K[0]

    def generate_report(self, df):
        """生成分析报告"""
        logging.info("生成分析报告中...")
        try:
            # 生成聚类报告
            cluster_report = {}
            for cid in df["cluster"].unique():
                cluster_df = df[df["cluster"] == cid]

                # 合并文本内容
                all_text = " ".join(cluster_df["cleaned"])

                report = {
                    "样本数量": len(cluster_df),
                    "代表性关键词": self.extract_keywords(all_text, 10),
                    "典型评论": self.extract_summary(all_text, 3),
                }
                cluster_report[f"聚类{cid}"] = report

            # 可视化分析
            self.visualize_results(df, cluster_report)

            # 保存结果
            with pd.ExcelWriter(self.config["output_file"]) as writer:
                df.to_excel(writer, sheet_name="详细数据")

                report_df = pd.DataFrame(cluster_report).T
                report_df.to_excel(writer, sheet_name="聚类分析")

                # 生成关键词云
                self.generate_wordclouds(cluster_report, writer)
        except Exception as e:
            logging.error(f"报告生成失败: {e}")
            raise

    def visualize_results(self, df, cluster_report):
        """可视化分析结果"""
        # 聚类分布可视化
        plt.figure(figsize=(10, 6))
        df["cluster"].value_counts().plot(kind="bar")
        plt.title("评论聚类分布", fontproperties=font)
        plt.xlabel("聚类编号",fontproperties=font)
        plt.ylabel("评论数量",fontproperties=font)
        plt.savefig(f"cluster_distribution_{label}.png")

        # 关键词云生成
        for cid, report in cluster_report.items():
            text = " ".join(report["代表性关键词"])
            wordcloud = WordCloud(
                font_path=font_path,
                width=800,
                height=400,
                background_color="white"
            ).generate(text)

            plt.figure()
            plt.imshow(wordcloud)
            plt.axis("off")
            plt.title(f"{cid} 关键词云", fontproperties=font)
            plt.savefig(f"wordcloud_{cid}_comments_{label}.png")
            plt.close()

    def generate_wordclouds(self, cluster_report, writer):
        """将词云插入Excel"""
        wb = writer.book
        ws = wb.create_sheet("关键词云")

        for idx, cid in enumerate(cluster_report.keys()):
            img = Image(f"wordcloud_{cid}_comments_{label}.png")
            ws.add_image(img, anchor=f"A{idx * 20 + 1}")


# ======================
# 执行主程序
# ======================
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    args = parser.parse_args()

    # 加载配置
    CONFIG = DEFAULT_CONFIG
    if args.config:
        try:
            import json
            with open(args.config, "r", encoding="utf-8") as f:
                CONFIG.update(json.load(f))
        except Exception as e:
            logging.warning(f"配置文件加载失败: {e}")

    # 启用进度条
    tqdm.pandas()

    # 执行分析
    analyzer = CommentAnalyzer(CONFIG)
    analyzer.process_file()

    logging.info(f"分析完成！结果已保存至: {CONFIG['output_file']}")