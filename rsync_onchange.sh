#!/bin/sh


# onchange is npm module
onchange -d 2000 './**/*' -- rsync -r -P --delete --exclude=".*" --exclude="code/profiles" --exclude="thesis" --exclude="code/data" --exclude="__pycache__" ~/dev/bachelor-thesis/ s8979104@taurusexport.hrsk.tu-dresden.de:/lustre/ssd/ws/s8979104-horovod/sync