#!/bin/sh


# onchange is npm module
onchange -d 2000 './**/*' -- rsync -r -P --delete --exclude=".*" --exclude="src/profiles" --exclude="thesis" --exclude="src/data" --exclude="__pycache__" ~/dev/bachelor-thesis-code/ s8979104@taurusexport.hrsk.tu-dresden.de:/lustre/ssd/ws/s8979104-horovod/sync