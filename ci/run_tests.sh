#!/usr/bin/env bash

parallel --header : python {test_module} ::: test_module tests/test_*.py