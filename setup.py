from setuptools import setup, find_packages

setup(
    name='pandasqlite',
    version='0.1.0',
    author="Daniel Karl I. Weidele",
    author_email="daniel.karl@ibm.com",
    description="Text-2-SQL on Pandas Data Frames",
    url="https://github.com/IBM/pandasqlite",
    python_requires='>=3.9',
    packages=find_packages(
        include=['pandasqlite']
    ),
    install_requires=[
        "PyMySQL==1.1.1",
        "ibm_watsonx_ai==1.3.20",
        "sqlalchemy==2.0.38"
    ]
)

