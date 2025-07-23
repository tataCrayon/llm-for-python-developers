import hashlib
from pathlib import Path
import os


class IdUtil:
    @staticmethod
    def generate_id(file_path: Path) -> str:
        """
        生成一个唯一的 ID，仅处理带文件格式后缀的路径

        Args:
            file_path (Path): 文件路径

        Returns:
            str: 唯一的 ID
        """
        if not file_path.suffix:
            raise ValueError("传入的路径必须指向一个带文件格式后缀的文件")

        # 获取路径的所有部分
        path_parts = file_path.parts

        # 取最后两个部分，如果不足两个则取全部
        relevant_parts = path_parts[-2:] if len(path_parts) >= 2 else path_parts

        # 拼接成新的路径字符串
        relevant_path = os.path.join(*relevant_parts)
        if not relevant_path or relevant_path.strip() == "":
            raise ValueError("提取的路径不能为空字符串、None 或者只包含空白字符")
        hash_object = hashlib.sha256(relevant_path.encode('utf-8'))
        doc_id = hash_object.hexdigest()
        return doc_id
