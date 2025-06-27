"""
Setup configuration for AI Brain Python package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    with open(filename, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Core requirements
install_requires = [
    "pydantic>=2.0.0",
    "motor>=3.3.0",  # Async MongoDB driver
    "pymongo>=4.5.0",
    "numpy>=1.24.0",
    "asyncio-throttle>=1.0.2",
    "python-dateutil>=2.8.2",
    "typing-extensions>=4.7.0",
    "psutil>=5.9.0",  # For system monitoring
]

# Optional framework dependencies
extras_require = {
    "crewai": [
        "crewai>=0.1.0",
    ],
    "pydantic-ai": [
        "pydantic-ai>=0.1.0",
    ],
    "agno": [
        "agno>=0.1.0",
    ],
    "langchain": [
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
    ],
    "langgraph": [
        "langgraph>=0.1.0",
    ],
    "all-frameworks": [
        "crewai>=0.1.0",
        "pydantic-ai>=0.1.0", 
        "agno>=0.1.0",
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "langgraph>=0.1.0",
    ],
    "dev": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "black>=23.7.0",
        "isort>=5.12.0",
        "mypy>=1.5.0",
        "flake8>=6.0.0",
        "pre-commit>=3.3.0",
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
    ],
    "docs": [
        "sphinx>=7.1.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
        "sphinx-autodoc-typehints>=1.24.0",
    ],
    "test": [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.21.0",
        "pytest-cov>=4.1.0",
        "pytest-mock>=3.11.0",
        "mongomock-motor>=0.0.21",
    ]
}

# Add 'all' extra that includes everything
extras_require["all"] = list(set(
    sum(extras_require.values(), [])
))

setup(
    name="ai-brain-python",
    version="0.1.0",
    author="AI Brain Team",
    author_email="team@aibrain.dev",
    description="Universal AI Brain with 16 cognitive systems and framework adapters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/romiluz13/AI_Brain",
    project_urls={
        "Bug Tracker": "https://github.com/romiluz13/AI_Brain/issues",
        "Documentation": "https://github.com/romiluz13/AI_Brain/blob/main/docs/API_REFERENCE.md",
        "Source Code": "https://github.com/romiluz13/AI_Brain",
    },
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Database :: Database Engines/Servers",
        "Typing :: Typed",
    ],
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require=extras_require,
    include_package_data=True,
    package_data={
        "ai_brain_python": ["py.typed"],
    },
    keywords=[
        "ai", "artificial-intelligence", "cognitive-systems", "agents", 
        "mongodb", "safety", "crewai", "pydantic-ai", "langchain", 
        "langgraph", "agno", "machine-learning", "nlp"
    ],
    entry_points={
        "console_scripts": [
            "ai-brain=ai_brain_python.cli:main",
        ],
    },
    zip_safe=False,
)
