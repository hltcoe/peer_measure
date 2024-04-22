import setuptools

with open('README.md', 'r') as f:
    long_description = f.read()

setuptools.setup(
    name='peer_measure',
    version='0.0.1',
    author='Eugene Yang',
    author_email='eugene.yang@jhu.edu',
    description="Implementation of the measure Probability of Equal Expected Rank",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/hltcoe/peer_measure',
    packages=setuptools.find_packages(),
    install_requires=["numpy", "scipy", "ir_measures"],
    include_package_data=True,
    python_requires='>=3.8',
)