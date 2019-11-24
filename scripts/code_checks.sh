#!/usr/bin/env bash
# Copyright 2019 Adam Gleave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# If you change these, also change .circle/config.yml.
SRC_FILES="src/ tests/ setup.py"
TYPECHECK_FILES="src/"  # tests often do weird things with types, exclude

set -x  # echo commands
set -e  # quit immediately on error

flake8 ${SRC_FILES}
black --check ${SRC_FILES}
codespell -I .codespell.skip --skip='*.pyc' ${SRC_FILES}

if [ -x "`which circleci`" ]; then
    circleci config validate
fi

if [ "$skipexpensive" != "true" ]; then
    pytype ${TYPECHECK_FILES}
    pylint -j 0 ${SRC_FILES}
fi
