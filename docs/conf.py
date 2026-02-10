project = "Axiom"
author = "Axiom Contributors"
release = "1.0.0"
version = "1.0"
copyright = "2026, Axiom Contributors"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "fieldlist",
    "deflist",
    "attrs_inline",
    "attrs_block",
]

myst_heading_anchors = 4

suppress_warnings = ["myst.xref_missing"]

html_theme = "pydata_sphinx_theme"

html_theme_options = {
    "github_url": "https://github.com/Frikallo/axiom",
    "show_toc_level": 2,
    "navigation_depth": 3,
    "use_edit_page_button": True,
}

html_context = {
    "github_user": "Frikallo",
    "github_repo": "axiom",
    "github_version": "main",
    "doc_path": "docs",
}
