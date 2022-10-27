#!/bin/bash

while true
do
    echo $(date +%s%N | cut -b1-13) >> /packets.log
    echo "$(ip -s link show dev lo)" >> /packets.log
    sleep $1
done
