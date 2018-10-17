#!/bin/bash

echo "wlan up"
sudo ifconfig wlan0 up
sleep 1

#echo "eth up"
#sudo ifconfig eth0 up
#sleep 1

sudo iw dev wlan0 link
sleep 1

echo "showing MAC address of connected router"
sudo ls /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/
sleep 1

#echo "showing rate table for IEEE 802.11 n"
#sudo cat /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/e8:94:f6:67:c6:ce/rate_scale_table
#sudo cat /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/04:f0:21:37:e7:94/rate_scale_table
#sleep 1

echo "selecting HT rate"
#echo 0x1c90e | sudo tee /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/e8:94:f6:67:c6:ce/rate_scale_table
#echo 0x1c90e | sudo tee /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/04:f0:21:37:e7:94/rate_scale_table
echo 0x1c90f | sudo tee /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/ac:84:c6:1c:81:0f/rate_scale_table
#sleep 1

#sudo echo 0x1c90e |sudo tee /sys/kernel/debug/ieee80211/phy0/iwlwifi/iwldvm/debug/monitor_tx_rate

# 0x420a = 1 Mbps
# 0x1890E = 6  Mbps
# 0x1c113 = 24 Mbps
# 0x1c90e = 54 Mbps

echo "verifying HT mode"
#sudo cat /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/04:f0:21:37:e7:94/rate_scale_table
#sudo cat /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/04:f0:21:37:e7:8e/rate_scale_table
sudo cat /sys/kernel/debug/ieee80211/phy0/netdev:wlan0/stations/ac:84:c6:1c:81:0f/rate_scale_table
#sleep 1

#AC:84:C6:1C:81:0F

#echo "enter option (1 = collect csi, 2 = livestream csi)"
#read number
#sleep 1

#if [ $number == '1' ]
#then
#	echo "collecting offline csi"
#	sleep 1

#	echo "enter sample name (e.g: test)"
#	read name

#	sleep 1
#	echo "collecting data"

#	echo "enter number of packets (e.g., 2000)"
#	read packets
#	sleep 1

#	sudo linux-80211n-csitool-supplementary/netlink/log_to_file $name.dat
#	sleep 1

#	sudo ping -i 0.0004 192.168.1.122 -c $packets
#	sleep 1
#fi


#echo "collecting offline csi"
#sleep 1

#echo "list data"
#ls
#sleep 1

echo "enter sample name (e.g: test)"
read name
sleep 1

#echo "collecting data"

sudo linux-80211n-csitool-supplementary/netlink/log_to_file $name.dat 
#sudo linux-80211n-csitool-supplementary/netlink/log_to_file $name.dat & tcpdump -i wlan0 -w $name.cap
sleep 1

#if [ $number == '2' ]
#then
#	echo "showing live plot"
#	cd linux-80211n-csitool-supplementary/netlink
#	sudo mv log_to_file.c log_to_file_orig.c
#	sudo mv log_to_file_livecsi.c log_to_file.c
#	sudo ./log
#fi

#shutdown() {
  # Get our process group id
#  PGID=$(ps -o pgid= $$ | grep -o [0-9]*)

  # Kill it in a new new process group
#  setsid kill -- -$PGID
#  exit 0
#}

#trap "shutdown" SIGINT SIGTERM
echo "bye"
exit
#kill -9 $PPID
