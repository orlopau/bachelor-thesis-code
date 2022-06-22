#!/bin/sh


# onchange is npm module
onchange -a './**/*.ipynb' -- jupytext --set-formats ipynb,py:percent --sync $PWD/{{file}}