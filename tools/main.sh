#!/usr/bin/sudo /bin/bash

echo "Enter option (1 = install kernel, 2 = install driver , 3 = make driver, 4 = verify driver, 5 = csi logging, 6 = enable network manager)"
read number
sleep 1

if [ $number == '1' ]
then
	echo "installing linux 3.13 kernel"
	sleep 1

	sudo apt-get update
	sleep 1

	sudo apt-get upgrade
	sleep 1

	sudo apt-get install linux-generic
	sleep 1

	sudo update-grub
	sleep 1

	sudo reboot
fi

if [ $number == '2' ]
then
	echo "installing csi drivers"
	sleep 1

	sudo apt-get install gcc make linux-headers-$(uname -r) git-core
	sleep 1

	CSITOOL_KERNEL_TAG=csitool-$(uname -r | cut -d . -f 1-2)
	sleep 1

	git clone https://github.com/dhalperi/linux-80211n-csitool.git
	sleep 1

	cd linux-80211n-csitool
	sleep 1

	git checkout ${CSITOOL_KERNEL_TAG}
	sleep 1

	make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi modules
	sleep 1

	sudo make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi INSTALL_MOD_DIR=updates modules_install
	sleep 1

	sudo depmod
	sleep 1

	cd ..
	sleep 1

	git clone https://github.com/dhalperi/linux-80211n-csitool-supplementary.git
	sleep 1

	for file in /lib/firmware/iwlwifi-5000-*.ucode; do sudo mv $file $file.orig; done
	sleep 1

	sudo cp linux-80211n-csitool-supplementary/firmware/iwlwifi-5000-2.ucode.sigcomm2010 /lib/firmware/
	sleep 1

	sudo ln -s iwlwifi-5000-2.ucode.sigcomm2010 /lib/firmware/iwlwifi-5000-2.ucode
	sleep 1

	make -C linux-80211n-csitool-supplementary/netlink
	sleep 1

	sudo reboot
fi

if [ $number == '3' ]
then
	echo "make driver"
	cd linux-80211n-csitool
	sleep 1

	sudo make -C /lib/modules/$(uname -r)/build M=$(pwd)/drivers/net/wireless/iwlwifi INSTALL_MOD_DIR=updates modules_install
	sleep 1

	sudo depmod
	sleep 1

	cd ..
	sleep 1
fi

if [ $number == '4' ]
then
	echo "verifying correct driver installation"
	sleep 1

	b=`iwconfig | grep -i wlan0 | wc -l`
	echo "$b"
	if [ $b == '1' ]
	then
	    echo "wifi is opened"
	elif [ $b == '0' ]
	then
	    echo "wifi is soft blocked"
	    exit 0
	fi
	sleep 1

	b=`iwconfig | grep -i Not-Associated | wc -l`
	echo "$b"
	if [ $b == '1' ]
	then
	    echo "You have not connected to the wifi ap"
	    exit 0
	elif [ $b == '0' ]
	then
	    echo "You have connected to the wifi AP"
	fi

	sleep 1

	a=`dmesg | grep -5 -i iwlwifi | grep connector_log | wc -l`
	if [ a == '1' ]
	then 
	    echo "driver is not OK!!! please reinstall"
	else
	    echo "driver is OK! Congratuation."
	fi
fi

if [ $number == '5' ]
then
	echo iface wlan0 inet manual | sudo tee -a /etc/network/interfaces
	sleep 1

	sudo ifconfig wlan0 up
	sleep 1

	#sudo iwconfig wlan0 essid neuralwave
	#sleep 1

	#sudo iw dev wlan0 link
	#sleep 1

	#sudo dhclient wlan0
	#sleep 1

	#sudo ping -i 0.001 192.168.1.122 -c 5
	#sleep 1

	echo "remove original driver"
	sudo modprobe -r iwlwifi mac80211
	sleep 1

	echo "enable csi logging"
	sudo modprobe iwlwifi connector_log=0x1
	sleep 1

	echo "installed modified driver"
	sleep 1

	sudo iwconfig wlan0 essid neuralwave
	sleep 1

	echo "connected to neuralwave"
#        sudo ifconfig wlan0 192.168.1.23 

	sudo dhclient wlan0
	sleep 1

	sudo ping -i 0.005 192.168.0.1 -c 30
fi

if [ $number == '6' ]
then
	echo "enabling network manager for internet"
	sleep 1

	sudo mv /etc/network/interfaces /etc/network/interfaces.bak
	sleep 1

	sudo /etc/init.d/networking restart
	sleep 1

	sudo restart network-manager
	sleep 1
fi
