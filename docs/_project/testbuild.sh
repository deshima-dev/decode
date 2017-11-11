#!/bin/bash

cat << EOS
INFO: this will build decode/testdocs
(this will not change any files of decode/docs)

EOS

if type sphinx-build >/dev/null 2>&1; then
    sphinx-apidoc -f -o ./apis ../../decode
    sphinx-build -a -d ../../testdocs/_doctree ./ ../../testdocs
    sphinx-build -a -d ../../testdocs/_doctree ./ ../../testdocs
else
    echo "ERROR: sphinx is not installed"
    exit 1
fi
