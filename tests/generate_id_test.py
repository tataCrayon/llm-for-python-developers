from pathlib import Path

from app.utils.id_util import IdUtil

print(IdUtil.generate_id(Path(r"F:/Projects/PythonProjects/llm-for-python-developers/tests/test.txt")))
print(IdUtil.generate_id(Path(r"F:/Projects/PythonProjects/llm-for-python-developers/tests/test.txt")))
print(IdUtil.generate_id(Path(r"F:/Projects/PythonProjects/llm-for-python-developers/test2/test.txt")))
print(IdUtil.generate_id(Path(r"F:/Projects/PythonProjects")))

