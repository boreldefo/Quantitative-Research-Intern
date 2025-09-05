# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

src_path = os.path.abspath('../..')
print(f"adding {src_path} in the path.")
sys.path.insert(0, src_path)  # so Sphinx finds modules one level up

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'A Mean Field Game of Shipping'
copyright = '2025, Charles-Albert Lehalle'
author = 'Charles-Albert Lehalle'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

html_theme = "sphinx_rtd_theme"

templates_path = ['_templates']
exclude_patterns = []

# Do not prepend module names to functions/classes
add_module_names = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_static_path = ['_static']
