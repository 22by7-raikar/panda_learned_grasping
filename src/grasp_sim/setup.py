from glob import glob

from setuptools import find_packages, setup

package_name = "grasp_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config/scenes", glob("config/scenes/*.yaml")),
        ("share/" + package_name + "/config", ["config/ycb_objects.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Anuj Pai Raikar",
    maintainer_email="anujpr240298@gmail.com",
    description="Synthetic RGB-D wrist camera and tabletop scenes for testing without hardware.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "synthetic_camera = grasp_sim.synthetic_camera_node:main",
            "render_scene = grasp_sim.scene:main",
        ],
    },
)
