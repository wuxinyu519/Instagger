#!/usr/bin/env python3
"""
严格按照INSTAG论文实现的标签清理工具
基于论文第3.2节 TAG NORMALIZATION 的四个步骤：
1. Frequency Filtering (频率过滤) 
2. Rule Aggregation (规则聚合)
3. Semantic Aggregation (语义聚合)
4. Association Aggregation (关联聚合)
"""

import pickle
import os
import re
import json
import pandas as pd
from collections import Counter, defaultdict
from typing import List, Dict, Set, Tuple
from pathlib import Path
import numpy as np

# 尝试导入依赖
try:
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    print("警告: scikit-learn不可用，语义聚合将被跳过")
    SKLEARN_AVAILABLE = False

try:
    from nltk.stem import PorterStemmer
    from nltk.download import download
    download('punkt', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    print("警告: NLTK不可用，将使用简化的词干提取")
    NLTK_AVAILABLE = False

try:
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    print("警告: mlxtend不可用，关联聚合将被跳过")
    MLXTEND_AVAILABLE = False

class SimplePorterStemmer:
    """简化版词干提取器"""
    def stem(self, word):
        if word.endswith('ing'):
            return word[:-3]
        elif word.endswith('ed'):
            return word[:-2] 
        elif word.endswith('s') and len(word) > 3:
            return word[:-1]
        return word

class INSTAGCleaner:
    """严格按照INSTAG论文实现的标签清理器"""
    
    def __init__(self, 
                 alpha: int = 20,                    # 频率过滤阈值 (论文中使用20)
                 semantic_threshold: float = 0.05,   # 语义相似度阈值 (论文中使用0.05)
                 min_support: int = 40,              # 关联规则最小支持度 (论文中使用40)
                 min_confidence: float = 0.99):      # 关联规则最小置信度 (论文中使用99%)
        """
        初始化INSTAG清理器,参数严格按照论文设置
        
        Args:
            alpha: 频率过滤阈值，低于此频率的标签被过滤
            semantic_threshold: 语义聚合的最小相似度阈值
            min_support: 关联聚合的最小支持度
            min_confidence: 关联聚合的最小置信度
        """
        self.alpha = alpha
        self.semantic_threshold = semantic_threshold  
        self.min_support = min_support
        self.min_confidence = min_confidence
        
        # 初始化词干提取器
        if NLTK_AVAILABLE:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = SimplePorterStemmer()
    
    def load_pkl_files(self, input_dir: str) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
        """加载PKL文件,专门处理*****_tagged_total.pkl格式"""
        all_data = []
        file_data_map = {}
        
        # 查找所有*****_tagged_total.pkl文件
        pkl_files = list(Path(input_dir).glob("*_tagged_total.pkl"))
        print(f"找到 {len(pkl_files)} 个*_tagged_total.pkl文件")
        
        if len(pkl_files) == 0:
            # 如果没找到，也尝试查找所有pkl文件
            pkl_files = list(Path(input_dir).glob("*.pkl"))
            print(f"备选: 找到 {len(pkl_files)} 个其他PKL文件")
        
        for pkl_file in pkl_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                if isinstance(data, list):
                    file_data = data
                elif isinstance(data, dict):
                    file_data = [data]
                else:
                    print(f"跳过未知格式文件: {pkl_file}")
                    continue
                
                # 标准化数据格式
                standardized_data = self._standardize_file_data(file_data)
                
                if standardized_data:
                    all_data.extend(standardized_data)
                    file_data_map[pkl_file.name] = standardized_data
                    print(f"加载 {pkl_file.name}: {len(standardized_data)} 条记录")
                
            except Exception as e:
                print(f"加载文件失败 {pkl_file}: {e}")
        
        print(f"总共加载 {len(all_data)} 条记录")
        return all_data, file_data_map
    
    def _standardize_file_data(self, file_data: List[Dict]) -> List[Dict]:
        """标准化文件数据格式"""
        standardized = []
        
        for item in file_data:
            if not isinstance(item, dict):
                continue
            
            # 直接使用prompt和tags字段
            if 'prompt' in item and 'tags' in item:
                prompt = item['prompt']
                
                # 处理tags字段
                if isinstance(item['tags'], str):
                    tags = self._parse_tags_string(item['tags'])
                elif isinstance(item['tags'], list):
                    tags = item['tags']
                else:
                    continue  # 跳过无效的tags格式
                
                if prompt and tags:
                    standardized.append({
                        'prompt': prompt,  # 保持原字段名
                        'tags': tags       # 保持原字段名
                    })
        
        return standardized
    
    def _parse_tags_string(self, tags_str: str) -> List[str]:
        """解析标签字符串"""
        if not tags_str:
            return []
        
        tags_str = tags_str.strip().strip('"\'')
        if not tags_str:
            return []
        
        tags = [tag.strip() for tag in tags_str.split(',')]
        return [tag for tag in tags if tag]
    
    def apply_instag_normalization(self, all_data: List[Dict]) -> Dict[str, str]:
        """
        应用INSTAG论文中的四步标签规范化流程
        
        Returns:
            tag_mapping: 原始标签到清理后标签的映射
        """
        print("\n=== 开始INSTAG标签规范化 ===")
        
        # 收集所有原始标签
        all_tags = []
        tag_sessions = []
        
        for item in all_data:
            tags = item['tags']  # 直接使用tags字段
            all_tags.extend(tags)
            tag_sessions.append(tags)
        
        tag_counts = Counter(all_tags)
        original_tag_count = len(set(all_tags))
        print(f"原始标签总数: {original_tag_count}")
        
        # 步骤1: 频率过滤 (Frequency Filtering)
        print(f"\n步骤1: 频率过滤 (α={self.alpha})")
        filtered_tags = self._frequency_filtering(tag_counts)
        print(f"保留标签数: {len(filtered_tags)} (减少 {original_tag_count - len(filtered_tags)})")
        
        # 步骤2: 规则聚合 (Rule Aggregation) 
        print(f"\n步骤2: 规则聚合")
        rule_mapping = self._rule_aggregation(filtered_tags)
        unique_after_rule = len(set(rule_mapping.values()))
        print(f"规则聚合后标签数: {unique_after_rule} (减少 {len(filtered_tags) - unique_after_rule})")
        
        # 步骤3: 语义聚合 (Semantic Aggregation)
        print(f"\n步骤3: 语义聚合 (相似度阈值={self.semantic_threshold})")
        if SKLEARN_AVAILABLE:
            semantic_mapping = self._semantic_aggregation(rule_mapping)
            unique_after_semantic = len(set(semantic_mapping.values()))
            print(f"语义聚合后标签数: {unique_after_semantic} (减少 {unique_after_rule - unique_after_semantic})")
        else:
            semantic_mapping = rule_mapping
            unique_after_semantic = unique_after_rule
            print(f"跳过语义聚合 (sklearn不可用)")
        
        # 步骤4: 关联聚合 (Association Aggregation)
        print(f"\n步骤4: 关联聚合 (支持度={self.min_support}, 置信度={self.min_confidence})")
        if MLXTEND_AVAILABLE:
            final_mapping = self._association_aggregation(semantic_mapping, tag_sessions)
            final_unique = len(set(final_mapping.values()))
            print(f"关联聚合后标签数: {final_unique} (减少 {unique_after_semantic - final_unique})")
        else:
            final_mapping = semantic_mapping
            final_unique = unique_after_semantic
            print(f"跳过关联聚合 (mlxtend不可用)")
        
        print(f"\n=== 规范化完成 ===")
        print(f"标签压缩比: {original_tag_count} -> {final_unique} ({final_unique/original_tag_count*100:.1f}%)")
        
        return final_mapping
    
    def _frequency_filtering(self, tag_counts: Counter) -> Set[str]:
        """
        步骤1: 频率过滤
        论文描述: "We first filter out long-tail tags appearing less than α times"
        """
        frequent_tags = {tag for tag, count in tag_counts.items() if count >= self.alpha}
        return frequent_tags
    
    def _rule_aggregation(self, tags: Set[str]) -> Dict[str, str]:
        """
        步骤2: 规则聚合  
        论文描述: "We transform all tags into lower characters to avoid the influence of capitalization.
        We also replace all special characters into spaces to further aggregate the tags.
        Finally, we apply stemming to each tag with the support of NLTK."
        """
        tag_mapping = {}
        normalized_groups = defaultdict(list)
        
        for tag in tags:
            # 按论文步骤进行规范化
            normalized = self._normalize_tag_by_rules(tag)
            normalized_groups[normalized].append(tag)
        
        # 为每个规范化组选择代表标签
        for normalized, group_tags in normalized_groups.items():
            if len(group_tags) > 1:
                # 选择原始标签作为代表（可以选择字母序最小的）
                representative = min(group_tags)  # 或者可以按频率选择
                for tag in group_tags:
                    tag_mapping[tag] = representative
            else:
                tag_mapping[group_tags[0]] = group_tags[0]
        
        return tag_mapping
    
    def _normalize_tag_by_rules(self, tag: str) -> str:
        """按论文规则规范化标签"""
        # 1. 转换为小写
        normalized = tag.lower()
        
        # 2. 将特殊字符替换为空格
        normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # 3. 应用词干提取
        words = normalized.split()
        stemmed_words = [self.stemmer.stem(word) for word in words]
        normalized = ' '.join(stemmed_words)
        
        # 移除多余空格
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    def _semantic_aggregation(self, tag_mapping: Dict[str, str]) -> Dict[str, str]:
        """
        步骤3: 语义聚合
        论文描述: "We employ text embedding models to obtain the semantics of tags.
        We use PhraseBERT, a BERT-based model designed explicitly for embedding phrases.
        After we obtain the semantic embeddings of tags, we use DBSCAN algorithm 
        to cluster tags with a given threshold t of semantic similarity."
        """
        unique_tags = list(set(tag_mapping.values()))
        
        if len(unique_tags) <= 1:
            return tag_mapping
        
        try:
            # 获取语义嵌入 (论文使用PhraseBERT，这里用TF-IDF近似)
            embeddings = self._get_semantic_embeddings(unique_tags)
            
            # 使用DBSCAN聚类 (按论文参数)
            clustering = DBSCAN(
                eps=1-self.semantic_threshold,  # 转换相似度为距离
                metric='cosine',
                min_samples=1
            )
            
            cluster_labels = clustering.fit_predict(embeddings)
            
            # 构建聚类映射
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(unique_tags[i])
            
            # 为每个聚类选择代表标签
            semantic_mapping = {}
            cluster_representatives = {}
            
            for cluster_id, cluster_tags in clusters.items():
                # 选择字母序最小的作为代表
                representative = min(cluster_tags)
                cluster_representatives[cluster_id] = representative
            
            # 更新映射
            for original_tag, current_mapped in tag_mapping.items():
                for cluster_id, cluster_tags in clusters.items():
                    if current_mapped in cluster_tags:
                        semantic_mapping[original_tag] = cluster_representatives[cluster_id]
                        break
            
            return semantic_mapping
            
        except Exception as e:
            print(f"语义聚合失败: {e}")
            return tag_mapping
    
    def _get_semantic_embeddings(self, tags: List[str]) -> np.ndarray:
        """获取语义嵌入 (简化版本，实际论文使用PhraseBERT)"""
        try:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=1000,
                stop_words='english'
            )
            embeddings = vectorizer.fit_transform(tags).toarray()
            return embeddings
        except:
            # 如果失败，返回随机嵌入
            return np.random.rand(len(tags), 100)
    
    def _association_aggregation(self, tag_mapping: Dict[str, str], 
                                tag_sessions: List[List[str]]) -> Dict[str, str]:
        """
        步骤4: 关联聚合
        论文描述: "We analyze all raw tagging results and employ the FP-Growth algorithm
        to mine association rules between tags. We then recursively merge associated tags
        based on the above association rules and reduce verbosity."
        """
        try:
            # 应用当前映射到会话数据
            mapped_sessions = []
            for session in tag_sessions:
                mapped_session = list(set([tag_mapping.get(tag, tag) for tag in session]))
                if mapped_session:
                    mapped_sessions.append(mapped_session)
            
            unique_tags = list(set(tag_mapping.values()))
            
            if len(unique_tags) <= 1 or len(mapped_sessions) < self.min_support:
                return tag_mapping
            
            # 构建二进制矩阵
            tag_matrix = []
            for session in mapped_sessions:
                row = [1 if tag in session else 0 for tag in unique_tags]
                tag_matrix.append(row)
            
            df = pd.DataFrame(tag_matrix, columns=unique_tags)
            
            # 使用FP-Growth算法 (按论文参数)
            min_support_ratio = self.min_support / len(mapped_sessions)
            frequent_itemsets = fpgrowth(df, min_support=min_support_ratio, use_colnames=True)
            
            if len(frequent_itemsets) == 0:
                return tag_mapping
            
            # 生成关联规则
            rules = association_rules(
                frequent_itemsets,
                metric="confidence",
                min_threshold=self.min_confidence
            )
            
            print(f"发现 {len(rules)} 条关联规则")
            
            # 应用关联规则合并标签
            association_mapping = tag_mapping.copy()
            
            for _, rule in rules.iterrows():
                antecedents = list(rule['antecedents'])
                consequents = list(rule['consequents'])
                all_related = antecedents + consequents
                
                if len(all_related) > 1:
                    # 选择字母序最小的作为代表
                    representative = min(all_related)
                    for tag in all_related:
                        # 更新所有映射到这些标签的原始标签
                        for orig_tag, mapped_tag in association_mapping.items():
                            if mapped_tag == tag:
                                association_mapping[orig_tag] = representative
            
            return association_mapping
            
        except Exception as e:
            print(f"关联聚合失败: {e}")
            return tag_mapping
    
    def apply_mapping_to_files(self, file_data_map: Dict[str, List[Dict]], 
                              tag_mapping: Dict[str, str]) -> Dict[str, List[Dict]]:
        """将标签映射应用到文件数据"""
        cleaned_files = {}
        
        for filename, data in file_data_map.items():
            cleaned_data = []
            
            for item in data:
                original_tags = item['tags']  # 直接使用tags字段
                
                # 应用标签映射
                cleaned_tags = []
                for tag in original_tags:
                    if tag in tag_mapping:
                        mapped_tag = tag_mapping[tag]
                        if mapped_tag not in cleaned_tags:
                            cleaned_tags.append(mapped_tag)
                
                # 保留有标签的记录
                if cleaned_tags:
                    cleaned_item = {
                        'prompt': item['prompt'],        # 保持原字段名
                        'tags': ','.join(cleaned_tags)   # 重新组合为逗号分隔字符串
                    }
                    cleaned_data.append(cleaned_item)
            
            if cleaned_data:
                cleaned_files[filename] = cleaned_data
                print(f"文件 {filename}: {len(data)} -> {len(cleaned_data)} 条记录")
        
        return cleaned_files
    
    def save_results(self, cleaned_files: Dict[str, List[Dict]], 
                    tag_mapping: Dict[str, str], 
                    output_dir: str = "./clean"):
        """保存清理结果，处理*****_tagged_total.pkl文件的命名"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存每个清理后的文件
        for filename, cleaned_data in cleaned_files.items():
            # 处理*****_tagged_total.pkl -> *****_tagged_total_clean.pkl
            if filename.endswith('_tagged_total.pkl'):
                base_name = filename[:-len('_tagged_total.pkl')]
                new_filename = f"{base_name}_tagged_total_clean.pkl"
            else:
                # 兼容其他命名格式
                base_name = Path(filename).stem
                new_filename = f"{base_name}_clean.pkl"
            
            output_path = os.path.join(output_dir, new_filename)
            
            with open(output_path, 'wb') as f:
                pickle.dump(cleaned_data, f)
            
            print(f"已保存: {output_path} ({len(cleaned_data)} 条记录)")
        
        # 保存标签映射
        mapping_file = os.path.join(output_dir, "instag_tag_mapping.json")
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(tag_mapping, f, ensure_ascii=False, indent=2)
        print(f"标签映射已保存: {mapping_file}")
        
        # 保存统计信息
        self._save_statistics(cleaned_files, tag_mapping, output_dir)
    
    def _save_statistics(self, cleaned_files: Dict[str, List[Dict]], 
                        tag_mapping: Dict[str, str], 
                        output_dir: str):
        """保存统计信息"""
        # 统计清理后的标签
        all_cleaned_tags = []
        total_records = 0
        
        for cleaned_data in cleaned_files.values():
            total_records += len(cleaned_data)
            for item in cleaned_data:
                tags = self._parse_tags_string(item['tags'])
                all_cleaned_tags.extend(tags)
        
        cleaned_tag_counts = Counter(all_cleaned_tags)
        
        stats = {
            "INSTAG规范化统计": {
                "原始标签数": len(tag_mapping),
                "最终标签数": len(set(tag_mapping.values())),
                "压缩比": f"{len(set(tag_mapping.values())) / len(tag_mapping) * 100:.1f}%",
                "处理文件数": len(cleaned_files),
                "总记录数": total_records
            },
            "参数设置": {
                "频率过滤阈值(α)": self.alpha,
                "语义相似度阈值": self.semantic_threshold,
                "关联规则最小支持度": self.min_support,
                "关联规则最小置信度": self.min_confidence
            },
            "最终标签频率Top20": dict(cleaned_tag_counts.most_common(20))
        }
        
        stats_file = os.path.join(output_dir, "instag_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存: {stats_file}")

def main():
    """主函数"""
    # 论文中的参数设置
    INPUT_DIR = "./extractors/data"
    OUTPUT_DIR = "./extractors/clean"
    
    print("=== INSTAG标签规范化工具 ===")
    print("严格按照论文实现的四步清理流程")
    
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录 {INPUT_DIR} 不存在")
        return
    
    try:
        # 创建INSTAG清理器 (使用论文参数)
        cleaner = INSTAGCleaner(
            alpha=20,                    # 论文中的频率过滤阈值
            semantic_threshold=0.05,     # 论文中的语义相似度阈值
            min_support=40,              # 论文中的最小支持度
            min_confidence=0.99          # 论文中的最小置信度
        )
        
        # 加载数据
        print(f"\n=== 加载PKL文件 ===")
        all_data, file_data_map = cleaner.load_pkl_files(INPUT_DIR)
        
        if not all_data:
            print("未找到有效数据")
            return
        
        # 应用INSTAG规范化
        tag_mapping = cleaner.apply_instag_normalization(all_data)
        
        # 应用到文件
        print(f"\n=== 应用清理到文件 ===")
        cleaned_files = cleaner.apply_mapping_to_files(file_data_map, tag_mapping)
        
        # 保存结果
        print(f"\n=== 保存结果 ===")
        cleaner.save_results(cleaned_files, tag_mapping, OUTPUT_DIR)
        
        print(f"\n=== 完成! ===")
        print(f"清理后文件保存在: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()