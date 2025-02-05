from setuptools import setup, find_packages

setup(
    name="conRWKV",
    version="0.1.0",  
    description="conRWKV",
    author="00ffcc",  
    author_email="guizhiyu@mail.ustc.edu.cn",
    packages=["conRWKV"],
    package_dir={"conRWKV": "conRWKV"},
    install_requires=[
        'fastapi',
        'uvicorn',
        'fla @ git+https://github.com/00ffcc/flash-linear-attention.git@main#egg=fla',
    ],
    entry_points={
        'console_scripts': [
            'conRWKV=conRWKV.main:main'
        ],
    },
    include_package_data=True, 
)
