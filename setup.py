from setuptools import find_packages, setup

setup_params = dict(
    name="worm_poses",
    description="Worm Segmentation",
    author="Avelino Javer",
    author_email="ver228@gmail.com",
    keywords=[],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)


def main() -> None:
    setup(**setup_params)


if __name__ == "__main__":
    main()
