#!/bin/bash
for((i=1;i<98;i+=1)); do var=$(printf "%03d" $i); montage /tmp/fframe$var.png /tmp/frame$var.png -geometry +0+0 -tile 1x2 /tmp/output_$var.png; done

