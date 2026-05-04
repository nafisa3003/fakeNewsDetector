import subprocess
import sys

subprocess.check_call([sys.executable, "-m", "pip", "install",
    "pandas", "numpy", "seaborn", "matplotlib", "nltk", "scikit-learn"])

subprocess.check_call([sys.executable, "-m", "pip", "install", "ipywidgets"])