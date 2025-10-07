from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="recon",
    version="0.1.0",
    author="Your Lab Name",
    author_email="your-email@university.edu",
    description="RECON: Efficient Multi-Hop RAG via Learned Context Compression",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/allfornancy/searchsum",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "vllm>=0.3.0",
        "ray>=2.9.0",
        "hydra-core>=1.3.0",
        "faiss-gpu>=1.7.0",
        "sentence-transformers>=2.2.0",
        "fastapi",
        "uvicorn",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
        ],
        "wandb": [
            "wandb>=0.16.0",
        ],
    },
)
