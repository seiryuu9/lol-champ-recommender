from setuptools import setup, find_packages

setup(
    name="lol-champ-recommender",
    version="0.1.0",
    description="League of Legends champion recommender using PyTorch",
    author="seiryuu9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ],
)
