# DEPENDENCIES
from setuptools import setup
from setuptools import find_packages


# Read the long description from README.md if it exists

readme_path = "README.md"

try:
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()

except FileNotFoundError:
    long_description = "AI-Powered Contract Risk Analysis Platform"

setup(name                          = "ai-contract-analyzer",
      version                       = "1.0.0",
      author                        = "Satyaki Mitra",
      author_email                  = "satyaki.mitra93@gmail.com",
      description                   = "An AI-powered platform for analyzing legal contracts and identifying potential risks.",
      long_description              = long_description,
      long_description_content_type = "text/markdown",
      url                           = "https://github.com/yourusername/ai-contract-analyzer", # Replace with your repo URL
      packages                      = find_packages(exclude = ["tests", "tests.*", "notebooks", "notebooks.*"]), # Exclude test and notebook directories
      classifiers                   = ["Development Status :: 4 - Beta",
                                       "Intended Audience :: Legal",
                                       "License :: OSI Approved :: MIT License", # Or your chosen license
                                       "Operating System :: OS Independent",
                                       "Programming Language :: Python :: 3",
                                       "Programming Language :: Python :: 3.10",
                                       "Programming Language :: Python :: 3.11",
                                      ],
      python_requires               = ">=3.10", # Based on modern library usage
      install_requires              = ["fastapi>=0.104.1",
                                       "uvicorn[standard]>=0.24.0",
                                       "pydantic>=2.5.0",
                                       "pydantic-settings>=2.1.0",
                                       "python-multipart>=0.0.6",
                                       "torch>=2.1.0",
                                       "transformers>=4.35.0",
                                       "sentence-transformers>=2.2.2",
                                       "tokenizers>=0.14.0",
                                       "safetensors>=0.4.0",
                                       "accelerate>=0.24.0",
                                       "numpy>=1.24.0",
                                       "pandas>=2.1.0",
                                       "scipy>=1.11.0",
                                       "spacy>=3.7.0",
                                       "reportlab>=4.0.0",
                                       "Pillow>=10.0.0",
                                       "PyPDF2>=3.0.0",
                                       "python-docx>=1.1.0",
                                       "requests>=2.31.0",
                                       "structlog>=23.1.0",
                                       "tqdm>=4.66.0",
                                       "python-dateutil>=2.8.0",
                                       "typing-extensions>=4.8.0",
                                       "anyio>=3.7.0",
                                       "psutil>=5.9.5",
                                      ],
      extras_require                = {"dev"      : ["black>=23.10.0", "isort>=5.12.0", "flake8>=6.0.0", "pytest>=7.4.0"],
                                       "openai"    : ["openai>=1.0.0"], # Optional OpenAI support
                                       "anthropic" : ["anthropic>=0.5.0"], # Optional Anthropic support
                                       "pymupdf"   : ["PyMuPDF>=1.23.0"], # Optional PyMuPDF support
                                      },
      entry_points                  = {"console_scripts": ["ai-contract-analyzer=app:main"]},
      include_package_data          = True,
     )
