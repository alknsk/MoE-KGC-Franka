from setuptools import setup, find_packages

# 读取README文件作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# 读取requirements文件，过滤掉注释行
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# 配置包信息
setup(
    name="moe-kgc-franka",
    version="0.1.0",
    author="Yang Haochen Sun Jinghan Yan jingbin",
    author_email="yanghc23@mails.tsinghua.edu.cn",
    description="用于Franka机器人人机交互中知识图谱构建的专家混合模型",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/moe-kgc-franka",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8", "mypy", "isort"],
        "gpu": ["torch-sparse", "torch-scatter", "torch-cluster", "torch-spline-conv"],
    },
    entry_points={
        "console_scripts": [
            "moe-kgc-train=scripts.train:main",
            "moe-kgc-evaluate=scripts.evaluate:main",
            "moe-kgc-demo=scripts.demo:main",
        ],
    },
)