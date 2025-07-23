# 使用 Python 基础镜像
# FROM 指令指定基础镜像。我们选择一个精简版的 Python 3.13 镜像。
FROM python:3.13.3-slim

# 设置工作目录
# WORKDIR 指令设置容器中的当前工作目录，后续的指令都将在此目录下执行。
WORKDIR /app

# 复制 requirements.txt 文件到工作目录
# 这一步是为了利用 Docker 的构建缓存。如果 requirements.txt 没有变化，
# Docker 将会使用缓存层，跳过重新安装依赖的步骤，从而加快构建速度。
COPY requirements.txt .

# 安装 Python 依赖
# RUN 指令在容器中执行命令。
# pip install --no-cache-dir -r requirements.txt 安装所有必要的 Python 库。
# --no-cache-dir 选项可以防止 pip 缓存下载的包，有助于减小最终镜像的大小。
RUN pip install --no-cache-dir -r requirements.txt

# 复制你的 FastAPI 应用程序的所有代码到容器中
# COPY . . 会将你本地 Dockerfile 所在目录下的所有文件和文件夹，复制到容器的 /app 目录下。
# 这一步放在依赖安装之后，可以确保只有当你的应用程序代码发生变化时，这一层才需要重新构建，
# 而不是每次代码变动都重新安装依赖。
COPY . .

# 暴露端口
# EXPOSE 指令声明容器会监听的端口。这仅仅是一个声明，用于文档说明，并不会实际发布端口。
# FastAPI 默认在 8000 端口运行。
EXPOSE 8000

# 定义容器启动时要执行的命令
# CMD 指令提供容器的默认启动命令。如果 docker run 命令提供了其他命令，CMD 将会被覆盖。
# 这里使用 uvicorn 启动你的 FastAPI 应用程序：
# - "uvicorn": 启动 ASGI 服务器。
# - "main:app": 指定 FastAPI 应用程序的入口。因为 main.py 在根目录，所以直接是 main:app。
#               `main` 指的是 `main.py` 文件，`app` 指的是你在 `main.py` 中创建的 FastAPI 实例（例如 `app = FastAPI()`）。
# - "--host 0.0.0.0": 让 Uvicorn 监听所有可用的网络接口，这样你可以从容器外部访问你的应用。
# - "--port 8000": 指定 Uvicorn 监听的端口。
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]