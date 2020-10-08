#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$( dirname "${SCRIPT_DIR}" )"

OPTIONS="-v -z -r -lpt"
EXCLUDES="LICENSE README.md CONTRIBUTING.md setup.py scripts/ .circleci/ examples/epic_demo.ipynb notebooks/ runners/launch_docker.sh tests/data/ .git *.pkl requirements*.txt"

# Refuse to compile if we find any of these words in non-excluded sources
# Adam(^|[^O]): excludes my name (Adam), but not AdamOptimizer
BLACKLISTED="Adam(^|[^O]) Gleave Leike Shane Legg Stuart Russell berkeley humancompatibleai humancompatible deepmind Google"

TMPDIR=`mktemp --tmpdir -d doubleblinded.XXXXXXXX`

SYNC_CMD="rsync ${OPTIONS} --exclude-from=.gitignore"
for exclude in ${EXCLUDES}; do
  SYNC_CMD="${SYNC_CMD} --exclude=${exclude}"
done

${SYNC_CMD} ${ROOT_DIR} ${TMPDIR}
pushd ${TMPDIR}

find . -type f | xargs -n 1 sed -i 's/# Copyright .*/# Copyright Anonymous Authors/'
find . -name '*.py' -o -name '*.sh' | xargs -n 1 sed -i -e 's/.*# TODO(.*).*//' -e 's/.*# SOMEDAY(.*).*//'

GREP_TERMS=""
for pattern in ${BLACKLISTED}; do
  GREP_TERMS="${GREP_TERMS} -e ${pattern}"
done
grep -r . -i -E ${GREP_TERMS}
if [[ $? -ne 1 ]]; then
  echo "Found blacklisted word. Dieing."
  exit 1
fi

rm ${ROOT_DIR}/supplementary.zip
zip -r ${ROOT_DIR}/supplementary.zip .
popd
