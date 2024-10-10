Getting Started
===============

Setup
-----

Requirements
~~~~~~~~~~~~

They shall be installed in your custom Python 3.10 environment using the usual set of commands:

::

    >>> conda create -n my_custom_env python=3.10 anaconda
    >>> conda activate my_custom_env
    >>> pip install -r requirements.txt

Make sure to always work in your custom environment!

Installation
~~~~~~~~~~~~

To get a runnable version of the code simply simply clone the repo using ``git`` and navigate to the
branch you need. By convention, ``master`` branch is kept (as much as possible) bug-free and is
updated when a new version of the tool is available. ``dev`` branch contains development work that
is to be released into a new version.

Development
-----------

Testing
~~~~~~~

We are using `pytest <https://docs.pytest.org/en/latest/>`__ as ourmain unit-testing framework.
To run all the unit-tests:

::

    >>> pytest

All unit-tests are located in ``tests/unit_tests`` and the folder/modules/class hierarchy fully
mimics the one of the source directory. These test architecture and naming conventions make it easy
understanding what test is testing what. To run a specific test which tests ``my_function``:

::

    >>> pytest tests/unit_tests/<my_module>/<my_script>.py::TestMyClass::test_my_function

Code coverage
~~~~~~~~~~~~~

| We are using
`coverage <https://coverage.readthedocs.io/en/v4.5.x/index.html>`__ as
| our code coverage tracker. To run the all the tests and track the
coverage:

::

    >>> coverage run --rcfile tests/.coveragerc -m pytest

Then, to print the coverage statistics:

::

    >>> coverage report --rcfile tests/.coveragerc


Documentation
~~~~~~~~~~~~~

We use
`sphinx <http://www.sphinx-doc.org/en/master/index.html>`__ as our documentation builder, in
conjunction with the following extensions:

-  **mathjax**: a parser for latex formulas
-  **autodoc**: an Sphinx extension for automatic API documentation

We use ReStructuredText as our primary documentation format. When documenting functions,
classes and modules, we respect the
`google style <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`__
docstring format.

To build the doc:

::

    >>> cd sphinx
    >>> make html

Documentation build is stored by default in ``sphinx/build/`` as a set of html pages, main page is
``index.html``.

Code style conventions
~~~~~~~~~~~~~~~~~~~~~~

There are a few conventions we try to respect as much as possible when
writing code in this project:

-  **Line length is 100 characters**
-  **We use Python 3.10 type hints**, for instance
   ``def func(arg1 : str, arg2: int) -> float``. See
   `here <https://docs.python.org/3/library/typing.html>`__ for more
   info
