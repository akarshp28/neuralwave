#!/bin/bash
while :
do
   sudo cat /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/ac:84:c6:1c:81:0f/rate_scale_table | grep 'MHz'
   sleep 1
done
