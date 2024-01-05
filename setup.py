import setuptools
import os

with open(os.path.join(os.path.dirname(__file__), "PAT", "version.txt"), "r", encoding="utf-8") as fv:
    __version__ = fv.read().strip()

with open("README.rst", "r", encoding="utf-8") as fd:
    long_description = fd.read()

with open("requirements.txt", "r", encoding="utf-8") as fr:
    requirements = [x.strip() for x in fr.readlines() if len(x.strip()) > 0]

setuptools.setup(
    name="PAT",
    version=__version__,
    author="RedRem",
    description="PodcastProject Analytics Toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license_files=('LICENSE',),
    url="https://github.com/GameOfPods/PAT",
    project_urls={
        "Bug Tracker": "https://github.com/GameOfPods/PAT/issues",
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        "License :: OSI Approved :: GPL-3.0-only",
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        'console_scripts': [
            'PAT = PAT.__main__:main',
        ],
    },
    packages=setuptools.find_packages(include=['PAT', 'PAT.*']),
    python_requires=">=3.7",
    install_requires=requirements,
    requires=requirements,
    include_package_data=True,
)
