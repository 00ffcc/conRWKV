from setuptools import setup, find_packages

setup(
    name="conRWKV",
    version="0.1.0",  
    description="conRWKV",
    author="00ffcc",  
    author_email="guizhiyu@mail.ustc.edu.cn",
    packages=find_packages(),
    install_requires=[
        'fastapi',
        'uvicorn',
        'flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention',
    ],
    entry_points={
        'console_scripts': [
            'conRWKV=conRWKV.main:main'
        ],
    },
    include_package_data=True, 
)
