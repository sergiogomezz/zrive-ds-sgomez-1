# 🚀 Zrive - Data Science Program

This repository contains the structure and guidelines for the **Zrive Applied Data Science** program:  
🔗 [https://zriveapp.com/cursos/zrive-applied-data-science](https://zriveapp.com/cursos/zrive-applied-data-science)

It is designed to maintain consistency, reproducibility, and best practices across all participants' projects. Inside, you'll find modular code, data access instructions, and standardized tooling for environment setup, testing, and code formatting.

---

## 🛠️ Environment Setup

To ensure consistent development environments, we use:

- **Python 3.11.0**, managed via [`pyenv`](https://github.com/pyenv/pyenv)
- **Virtual environments and dependencies** managed via [`poetry`](https://python-poetry.org/)

Other alternatives (e.g., `venv`, `pipenv`, `conda`) are valid, but for this course we recommend using `pyenv + poetry`.

> ⚠️ **IMPORTANT**: Never use the system-wide Python installation. Always isolate your project using a virtual environment to avoid conflicts between dependencies.

### 🔧 Step-by-step Setup

1. **Install Python 3.11.0**  
   Make sure `pyenv` is installed, then run:
   ```bash
   pyenv install 3.11.0
