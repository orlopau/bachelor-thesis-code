#!/bin/sh


# onchange is npm module
onchange -d 2000 './**/*' -- rsync -r -P --delete --exclude=".*" --exclude="thesis" --exclude="code/data" ~/dev/bachelor-thesis/ s8979104@taurusexport.hrsk.tu-dresden.de:/lustre/ssd/ws/s8979104-horovod/sync