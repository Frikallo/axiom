project = "Axiom"
author = "Axiom Contributors"

extensions = [
    "myst_parser",
    "breathe",
    "sphinx_copybutton",
    "sphinx_design",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- MyST --------------------------------------------------------------------
myst_enable_extensions = ["colon_fence"]

# -- Breathe (Doxygen XML → Sphinx) -----------------------------------------
breathe_projects = {"Axiom": "xml"}
breathe_default_project = "Axiom"
breathe_default_members = ("members", "undoc-members")

# -- Theme -------------------------------------------------------------------
html_theme = "furo"

pygments_style = "friendly"
pygments_dark_style = "github-dark"

html_theme_options = {
    "source_repository": "https://github.com/Frikallo/axiom",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#2962ff",
        "color-brand-content": "#2962ff",
    },
    "dark_css_variables": {
        "color-brand-primary": "#82b1ff",
        "color-brand-content": "#82b1ff",
    },
}

html_logo = "_static/axiomlogo.png"
html_favicon = "_static/axiomlogo.png"

html_static_path = ["_static"]
html_css_files = ["custom.css"]
