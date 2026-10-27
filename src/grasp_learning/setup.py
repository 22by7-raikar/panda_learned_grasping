from glob import glob

from setuptools import find_packages, setup

package_name = "grasp_learning"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anuj Pai Raikar",
    maintainer_email="anujpr240298@gmail.com",
    description="Learned grasp detection: multi-grasp Faster R-CNN (Chu et al. 2018) on RGD images.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "frcnn_grasp_node = grasp_learning.frcnn_grasp_node:main",
            "train = grasp_learning.train:main",
            "evaluate = grasp_learning.evaluate:main",
        ],
    },
)
