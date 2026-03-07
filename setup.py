"""Setup script for NETAI Predictive Analytics & Forecasting."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="netai-forecast",
    version="0.1.0",
    author="Ali",
    description="AI-powered predictive analytics and forecasting for network performance (NETAI / GSoC 2026)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/NETAI-Predictive-Analytics-Forecasting-Ali",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Networking :: Monitoring",
    ],
    entry_points={
        "console_scripts": [
            "netai-train=scripts.train:main",
            "netai-evaluate=scripts.evaluate:main",
        ],
    },
)
