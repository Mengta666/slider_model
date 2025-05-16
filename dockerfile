# 使用 debian:latest 作为基础镜像
FROM debian:latest

# 设置工作目录
WORKDIR /flask_app/slider_model

# 复制 slider_model.tar 到容器
COPY slider_model.tar /flask_app/slider_model/

# 安装必要工具和 Python 3.11
RUN apt-get update && \
    apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    tar \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 解压 slider_model.tar 并移动到上一级目录
RUN tar -xf slider_model.tar && \
    mv slider_model/* . || true && \
    ls -la /flask_app/slider_model/ && \
    ls -la /flask_app/slider_model/venv/ || echo "venv directory not found" && \
    rm slider_model.tar

# 检查虚拟环境，若缺失则重新创建并安装依赖
RUN if [ -f "venv/bin/activate" ]; then \
        . venv/bin/activate && \
        /flask_app/slider_model/venv/bin/pip install gunicorn; \
    else \
        python3.11 -m venv venv && \
        . venv/bin/activate && \
        /flask_app/slider_model/venv/bin/pip install \
            torch==2.3.1+cpu \
            torchvision==0.18.1+cpu \
	    --index-url https://download.pytorch.org/whl/cpu &&\
	/flask_app/slider_model/venv/bin/pip install\
            flask \
            pillow \
            numpy \
            gunicorn; \
    fi

# 确保必要目录存在
RUN mkdir -p Uploads static/output saved_images saved_coco_json && \
    chmod -R 755 Uploads static/output saved_images saved_coco_json

# 设置环境变量以确保 Python 使用虚拟环境
ENV PATH="/flask_app/slider_model/venv/bin:$PATH"

# 运行 Flask 应用
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "slider_model:slider_model_pre"]
