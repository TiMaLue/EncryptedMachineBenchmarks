#!/usr/bin/env bash
if [[ "$LO_ENABLE_MTU_LIMIT" == 1 ]]
    then
    echo "Setting localhost link mtu to $LO_MTU"
    ip link set dev lo mtu $LO_MTU
    ip link list
fi 

if [[ "$LO_ENABLE_TC_DEL" == 1  ]] && [[ "$LO_ENABLE_TC_RATE" == 1  ]]
then
    echo "Enabling TC del and rate on localhost: Setting delay to $LO_DELAY_MS ms and bandwidth to $LO_RATE_MBIT mbit"
    tc qdisc add dev lo root netem delay "$LO_DELAY_MS"ms rate "$LO_RATE_MBIT"mbit limit 100000
    tc -s qdisc ls dev lo
else
    if [[ "$LO_ENABLE_TC_DEL" == 1  ]] 
    then
        echo "Enabling TC del on localhost: Setting delay to $LO_DELAY_MS ms"
        tc qdisc add dev lo root netem delay "$LO_DELAY_MS"ms
        tc -s qdisc ls dev lo
    else
        if [[ "$LO_ENABLE_TC_RATE" == 1  ]] 
        then
            echo "Enabling TC rate on localhost: Setting bandwidth to $LO_RATE_MBIT mbit"
            tc qdisc add dev lo root netem rate "$LO_RATE_MBIT"mbit limit 100000
            tc -s qdisc ls dev lo
        fi
    fi
fi


/bin/watch_packets.bash $1
