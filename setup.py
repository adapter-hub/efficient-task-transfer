from setuptools import setup


setup(
    name="itrain",
    version="0.0",
    author="calpt",
    author_email="calpt@mail.de",
    description="Simple training setup",
    license="MIT",
    py_modules=["itrain"],
    install_requires=[
        "adapter-transformers == 1.1.1",
        "datasets == 1.6.2",
        "scikit-learn == 0.23.2",
        "seqeval == 1.2.2",
        "tgsend >= 0.3",
        "torch == 1.7.1",
        "tqdm",
        "yagmail",
        "keyring",
    ],
    entry_points={
        "console_scripts": ["itrain=itrain.itrain:main"]
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
