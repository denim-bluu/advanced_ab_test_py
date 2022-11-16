# Python Project Template

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
![Tests](https://github.com/indigo-bluu/bidenance/actions/workflows/tests.yml/badge.svg)

- [Python Project Template](#python-project-template)
  - [Scope](#scope)
  - [Structure](#structure)
  - [Github](#github)
    - [Branching Guideline](#branching-guideline)
    - [Commit Convention](#commit-convention)
    - [Python Virtual Environment Setting](#python-virtual-environment-setting)
      - [Build your virtual environment and install all packages](#build-your-virtual-environment-and-install-all-packages)
  - [Formatting (Preferred)](#formatting-preferred)
    - [Black formatting](#black-formatting)
  - [Pre-Commit (Optional)](#pre-commit-optional)
    - [Installation](#installation)
    - [Define pre-commit config file](#define-pre-commit-config-file)
    - [Adding additional hook plugins](#adding-additional-hook-plugins)
    - [Install git hook scripts](#install-git-hook-scripts)
    - [(Option) Run against all files](#option-run-against-all-files)
  - [Test Driven Development (Optional)](#test-driven-development-optional)
    - [pyest](#pyest)
    - [Implementation](#implementation)
    - [How to invoke pytest](#how-to-invoke-pytest)

## Scope

TBD

## Structure

TBD

## Github

### Branching Guideline

The branching structure in this development pipeline follows the branch-structure described in this page: [Branching](https://gist.github.com/digitaljhelms/4287848)
### Commit Convention

The commit convention is well described in this page: [Conventional Commit Messages](https://gist.github.com/qoomon/5dfcdf8eec66a051ecd85625518cfd13#types)

### Python Virtual Environment Setting

#### Build your virtual environment and install all packages

Run shell-script with Makefile

```zsh
make setup_venv
```

Or manually create a virtual environment with Python version ^3.10 and install all packages via `poetry`.
## Formatting (Preferred)

### Black formatting

All code in this repository should be `Black` formatted. Black is a very useful tool for formatting, it makes code review faster by producing the smallest diffs possible.
Blackened code looks the same regardless of the project you’re reading. Formatting becomes transparent after a while and you can focus on the content instead.
More on this here <https://black.readthedocs.io/en/latest/>.

Using Github actions (setup in black.yml) all code PRs are tested for black formatting to ensure consistency across the repo

## Pre-Commit (Optional)

[Git pre-commit hook](<(https://pre-commit.com/)>) feature is implemented to ensure that the repository is maintained
with consistent manner and automatically flag undetected errors & issues prior to code commit

### Installation

Prior to running hooks, need to install:

```zsh
pip install pre-commit
```

### Define pre-commit config file

Configuration file is needed to specify hook plugins that you want to add.
If you don't have any configuration file, then add `.pre-commit-config.yaml` in the root
repository. Once you have defined your hooks in the config file, they will run automatically
every time you commit

### Adding additional hook plugins

If you want to add additional hook plugins, simply add hook in `.pre-commit-config,yaml` file:

```yaml
# Example
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.0.1
    hooks:
      - id: trailing-whitespace
      - id: check-json
      - id: end-of-file-fixer
      - id: name-tests-test
      - id: requirements-txt-fixer

  - repo: https://github.com/psf/black
    rev: 19.3b0
    hooks:
      - id: black
```

Here is the link to the support hooks [Link](https://pre-commit.com/hooks.html)

### Install git hook scripts

run `pre-commit install` to set up the git hook scripts

```zsh
pre-commit install
```

### (Option) Run against all files

If you want to run pre-commit hook checks on all files in the project, type:

```zsh
pre-commit run --all-files
```

This code will run all pre-commit checks.

## Test Driven Development (Optional)

### pyest

pytest is one of the best tools for effective TDD framework.
Compared to built-in tool **unittest** module, pytest provides handful features that
could potentially enhance the TDD framework.

For detailed implementation of PyTest, please refer to the PyTest
[website](https://docs.pytest.org/en/6.2.x/).

### Implementation

Currently, pytest is amalgamated with Git Action so that it is invoked everytime the user
push or PR to the repository. The relevant Git Action YAML file is located
[here](.github/workflows/tests.yml).

If you want to invoke pytest locally, please refer to the quick examples below.
(For further details, please refer to the website)

### How to invoke pytest

Run tests in a module:

```zsh
pytest tests/unittest/testing_file.py
```

Run tests in a directory:

```zsh
pytest tests/unittest/
```

Run all tests (From root directory):

```zsh
pytest .
```
