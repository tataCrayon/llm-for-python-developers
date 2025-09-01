import json
import re
from pathlib import Path
from typing import List

from langchain.docstore.document import Document

from app.config.logger import setup_logger
from app.utils.id_util import IdUtil
from app.utils.keyword_extractor import KeywordExtractor

logger = setup_logger(__name__)


class MDProcessorConfig:
    """
    Markdown 处理器配置类
    """

    def __init__(
            self,
            chunk_strategy: str = "recursive",  # 分块策略：paragraph, heading, fixed_length ,recursive
            max_chunk_size: int = 1500,  # 固定长度分块的最大字符数
            chunk_overlap: int = 200,  # 相邻块之间的重叠字符数
            heading_level: int = 4,  # 按标题分块的最大标题级别（例如 h1-h4）
            min_chunk_size: int = 50  # 最小块大小
    ):
        self.chunk_strategy = chunk_strategy
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = min(chunk_overlap, max_chunk_size // 2)  # 确保重叠不超过块大小的一半
        self.heading_level = heading_level
        self.min_chunk_size = min_chunk_size
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")


class MDProcessor:
    """
    Markdown 文件处理器，支持多种分块策略
    """

    def __init__(self, config: MDProcessorConfig | None = None):
        """
        初始化 Markdown 处理器。
        Args:
            config 可选: 分块配置，默认为按段落分块。
        """
        self.config = config or MDProcessorConfig()
        logger.info(f"Initialized MDProcessor with strategy: {self.config.chunk_strategy}, "
                    f"max_chunk_size: {self.config.max_chunk_size}, "
                    f"chunk_overlap: {self.config.chunk_overlap}")

    def _split_by_heading_with_content(self, content: str) -> List[str]:
        """
        按标题分块，但确保每个块包含标题下的完整内容
        Args:
            content: 待分块的文本内容
        Returns:
            分块后的文本列表
        """
        lines = content.splitlines()
        chunks = []
        current_chunk = []
        current_title = ""

        for i, line in enumerate(lines):
            # 检测标题
            title_match = re.match(r'^(#{1,%d})\s+(.*)$' % self.config.heading_level, line)

            if title_match:
                # 保存前一个块（如果有足够内容）
                if current_chunk:
                    chunk_content = '\n'.join(current_chunk).strip()
                    if len(chunk_content) >= self.config.min_chunk_size:
                        chunks.append(chunk_content)

                # 开始新块
                current_chunk = [line]
                current_title = title_match.group(2)
            else:
                current_chunk.append(line)

            # 检查当前块是否过大
            current_text = '\n'.join(current_chunk)
            if len(current_text) > self.config.max_chunk_size and len(current_chunk) > 1:
                # 保留标题，分割内容
                title_line = current_chunk[0]
                content_lines = current_chunk[1:]

                if content_lines:
                    content_text = '\n'.join(content_lines)

                    # 尝试按段落分割
                    paragraphs = [p.strip() for p in content_text.split('\n\n') if p.strip()]

                    sub_chunks = []
                    current_sub = []
                    current_size = len(title_line) + 1  # +1 for newline

                    for para in paragraphs:
                        para_size = len(para) + 2  # +2 for double newline

                        if current_size + para_size <= self.config.max_chunk_size:
                            current_sub.append(para)
                            current_size += para_size
                        else:
                            # 保存当前子块
                            if current_sub:
                                sub_content = '\n\n'.join(current_sub)
                                sub_chunks.append(f"{title_line}\n\n{sub_content}")

                            # 开始新的子块
                            current_sub = [para]
                            current_size = len(title_line) + 1 + para_size

                    # 添加最后一个子块
                    if current_sub:
                        sub_content = '\n\n'.join(current_sub)
                        sub_chunks.append(f"{title_line}\n\n{sub_content}")

                    # 添加所有子块
                    for j, sub_chunk in enumerate(sub_chunks):
                        if j == 0:
                            chunks.append(sub_chunk)
                        else:
                            # 为续块添加标识
                            title_suffix = "## {title} (续{index})\n\n".format(title=current_title, index=j)
                            content_part = sub_chunk.split('\n\n', 1)[1] if '\n\n' in sub_chunk else sub_chunk
                            chunks.append(title_suffix + content_part)

                current_chunk = []

        # 添加最后一个块
        if current_chunk:
            chunk_content = '\n'.join(current_chunk).strip()
            if len(chunk_content) >= self.config.min_chunk_size:
                chunks.append(chunk_content)

        return chunks

    def _calculate_content_weight(self, chunk: str) -> float:
        """
        基于内容质量计算权重

        Args:
            chunk: 待计算权重的块
        Returns:
            块的权重
        """
        base_weight = 0.6

        # 标题权重
        if re.match(r'^#{1,3}\s+', chunk):
            base_weight += 0.4

        # 内容长度权重（避免过短的块）
        content_length = len(chunk.strip())
        if content_length > 200:
            base_weight += 0.2
        elif content_length < 50:
            base_weight -= 0.3

        # 包含代码块或实例
        if '```' in chunk or '示例：' in chunk or '例如：' in chunk:
            base_weight += 0.2

        # 包含列表或结构化信息
        if re.search(r'^\s*[-\*\+]\s+', chunk, re.MULTILINE) or re.search(r'^\s*\d+\.\s+', chunk, re.MULTILINE):
            base_weight += 0.1

        return min(base_weight, 1.0)

    def _split_by_paragraph(self, content: str) -> List[str]:
        """
        按段落分块
        """
        logger.info("Splitting by paragraph")
        return [p.strip() for p in content.split("\n\n") if p.strip()]

    # 存在问题，废弃
    def _split_by_heading(self, content: str) -> List[str]:
        """
        按标题分块（支持 h1 到指定级别的标题）
        """
        logger.info("Splitting by heading")
        heading_pattern = r'^(#{1,%d})\s+.*$' % self.config.heading_level
        chunks = []
        current_chunk = []
        for line in content.splitlines():
            if re.match(heading_pattern, line):
                if current_chunk:
                    chunks.append("\n".join(current_chunk).strip())
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)
        if current_chunk:
            chunks.append("\n".join(current_chunk).strip())
        return [c for c in chunks if c]

    def _split_by_fixed_length(self, content: str) -> List[str]:
        """
        按固定长度分块
        """
        logger.info("Splitting by fixed length")
        chunks = []
        step = self.config.max_chunk_size - self.config.chunk_overlap

        for i in range(0, len(content), step):
            end = min(i + self.config.max_chunk_size, len(content))
            chunk = content[i:end]

            # 在重叠区域尝试找到更好的分割点（段落边界）
            if end < len(content) and i > 0:
                # 寻找段落边界
                overlap_start = max(0, end - self.config.chunk_overlap)
                overlap_text = content[overlap_start:end + 50]  # 多看一点

                # 寻找双换行符（段落边界）
                para_break = overlap_text.find('\n\n')
                if para_break != -1:
                    new_end = overlap_start + para_break
                    if new_end > i:  # 确保不会产生空块
                        chunk = content[i:new_end + 2]  # 包含换行符
            chunks.append(chunk.strip())

        return [c for c in chunks if len(c) >= self.config.min_chunk_size]

    def _split_recursively(self, content: str) -> List[str]:
        """
        递归分块：先按标题，再按段落，最后按固定长度
        """
        logger.info("Splitting recursively")
        chunks = self._split_by_heading_with_content(content)
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= self.config.max_chunk_size:
                final_chunks.append(chunk)
            else:
                # 过大的块按段落分
                sub_chunks = self._split_by_paragraph(chunk)
                for sub_chunk in sub_chunks:
                    if len(sub_chunk) <= self.config.max_chunk_size:
                        final_chunks.append(sub_chunk)
                    else:
                        # 仍过大的块按固定长度分
                        final_chunks.extend(self._split_by_fixed_length(sub_chunk))
        return final_chunks

    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        合并过小的相邻块
        """
        if not chunks:
            return chunks

        merged_chunks = []
        current_chunk = chunks[0]

        for i in range(1, len(chunks)):
            next_chunk = chunks[i]

            # 如果当前块太小，尝试与下一块合并
            if (len(current_chunk) < self.config.min_chunk_size and
                    len(current_chunk + "\n\n" + next_chunk) <= self.config.max_chunk_size):
                current_chunk = current_chunk + "\n\n" + next_chunk
            else:
                # 保存当前块，开始处理下一块
                if len(current_chunk) >= self.config.min_chunk_size:
                    merged_chunks.append(current_chunk)
                current_chunk = next_chunk

        # 添加最后一块
        if len(current_chunk) >= self.config.min_chunk_size:
            merged_chunks.append(current_chunk)
        elif merged_chunks:
            # 如果最后一块太小，与前一块合并
            last_chunk = merged_chunks.pop()
            if len(last_chunk + "\n\n" + current_chunk) <= self.config.max_chunk_size:
                merged_chunks.append(last_chunk + "\n\n" + current_chunk)
            else:
                merged_chunks.extend([last_chunk, current_chunk])

        return merged_chunks

    def process(self, file_path: Path) -> List[Document]:
        """
        处理 Markdown 文件并生成 Document 对象。
        Args:
            file_path (Path): Markdown 文件路径。
        Returns:
            List[Document]: 分块后的 Document 对象列表。
        """
        logger.info(f"Start to process file: {file_path}")
        file_id = IdUtil.generate_id(file_path)

        try:
            with file_path.open('r', encoding='utf-8') as f:
                content = f.read()

            # 根据配置选择分块策略
            if self.config.chunk_strategy == "paragraph":
                chunks = self._split_by_paragraph(content)
            elif self.config.chunk_strategy == "heading":
                chunks = self._split_by_heading_with_content(content)
            elif self.config.chunk_strategy == "fixed_length":
                chunks = self._split_by_fixed_length(content)
            elif self.config.chunk_strategy == "recursive":
                chunks = self._split_recursively(content)
            else:
                logger.error(f"Unsupported chunk strategy: {self.config.chunk_strategy}")
                return []

            # 合并过小的块
            chunks = self._merge_small_chunks(chunks)

            # 提取关键词（为混合检索预留）
            kw_extractor = KeywordExtractor()
            documents = []

            for i, chunk in enumerate(chunks):
                # 基本质量检查
                if len(chunk.strip()) < self.config.min_chunk_size:
                    logger.debug(f"跳过过小的块: {chunk[:50]}...")
                    continue

                keyword_list = kw_extractor.extract_keywords(chunk)
                keywords = ','.join(keyword_list)

                # 获取上下文（前后文本片段）
                prev_context = chunks[i - 1][-100:] if i > 0 else ""
                next_context = chunks[i + 1][:100] if i < len(chunks) - 1 else ""

                # id= file_id_chunk_num
                doc_id = f"{file_id}_{i}"

                context = json.dumps({'prev': prev_context, 'next': next_context}, ensure_ascii=False)

                document = Document(
                    page_content=chunk,
                    metadata={
                        'id': doc_id,
                        'source': str(file_path),
                        'file_name': file_path.name,
                        'extension': file_path.suffix,
                        'chunk_index': i,
                        'chunk_strategy': self.config.chunk_strategy,
                        'chunk_overlap': self.config.chunk_overlap,
                        'keywords': keywords,
                        'context': context,
                        # 'weight': 1.0 if re.match(r'^#{1,3}\s+', chunk) else 0.8
                        'weight': self._calculate_content_weight(chunk),
                        'chunk_size': len(chunk)
                    }
                )
                logger.debug(f"The document after chunk: {document}")
                documents.append(document)
            logger.info(f"Processed {file_path}: {len(documents)} chunks")
            return documents
        except Exception as e:
            logger.error(f"Error processing Markdown file {file_path}: {e}")
            return []