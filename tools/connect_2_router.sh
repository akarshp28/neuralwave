sudo modprobe -r iwlwifi mac80211
sudo modprobe iwlwifi connector_log=0x1
sudo ifconfig wlan0 up
sudo iwconfig wlan0 essid TP-LINK_8110_5G
sudo dhclient wlan0
sudo ping -i 0.0004 192.168.0.1 -c 10




