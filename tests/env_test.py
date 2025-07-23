import os
from dotenv import load_dotenv
from pathlib import Path

# 1. 加载env文件
load_dotenv()

print(os.getenv('DEEPSEEK_API_KEY'))

print((Path("F:/ProgrammingEnvironment/AI/EmbeddingModel/BGE-Large-ZH") / "config.json").exists())

base = Path("F:/ProgrammingEnvironment/AI/EmbeddingModel")
for d in base.iterdir():
    if '\u200b' in d.name:
        new_name = d.name.replace('\u200b', '')
        print(f"Renaming: {d.name} -> {new_name}")
        d.rename(base / new_name)

for name in os.listdir("F:/ProgrammingEnvironment/AI/EmbeddingModel"):
    print(repr(name))



