![QimPy](docs/qimpy.svg) (with Effective Atom Theory)

---

QimPy (pronounced [/'kɪm-paɪ'/](https://en.wikipedia.org/wiki/Help:IPA/English))
is a Python package for Quantum-Integrated Multi-PhYsics. This repository implements
effective atom theory (EAT) in QimPy for rapid gradient-based 
optimization of objective functions involving converged DFT energies. The original code 
without EAT can be found here: https://github.com/shankar1729/qimpy.git.

# Coding style

The repository provides a .editorconfig with indentation and line-length rules,
and a pre-commit configuration to run black and flake8 to enforce and verify style.
Please install this pre-commit hook by running `pre-commit install`
within the working directory.
While this hook will run automatically on filed modified in each commit,
you can also use `make precommit` to manually run it on all code files. 

Function/method signatures and class attributes must use type hints.
Document class attributes using doc comments on the type hints when possible.
Run `make typecheck` to perform a static type check using mypy before pushing code.

For all log messages, use f-strings as far as possible for maximum readability.

Run `make test` to invoke all configured pytest tests. To only run mpi or
non-mpi tests specifically, use `make test-mpi` or `make test-nompi`.

Best practice: run `make check` to invoke the precommit, typecheck
and test targets before commiting code.
