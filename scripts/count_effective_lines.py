"""
此脚本用于统计指定工程目录下所有 Python 文件的有效代码行数，
有效代码行指非空行且不以注释符号开头的行。
"""
import os

def count_lines_in_file(file_path: str) -> int:
    """
    统计单个文件的有效代码行数。

    Args:
        file_path (str): 文件的路径。

    Returns:
        int: 有效代码行数。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            # 过滤掉空行和仅含注释的行
            return sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
    except Exception:
        return 0

def count_lines_in_directory(directory: str) -> int:
    """
    递归统计指定目录下所有 Python 文件的有效代码行数。

    Args:
        directory (str): 目录的路径。

    Returns:
        int: 所有 Python 文件的有效代码总行数。
    """
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                total_lines += count_lines_in_file(file_path)
    return total_lines

if __name__ == "__main__":
    project_directory = "F:/Projects/PythonProjects/llm-for-python-developers/app"
    effective_lines = count_lines_in_directory(project_directory)
    print(f"工程的有效代码行数: {effective_lines}")