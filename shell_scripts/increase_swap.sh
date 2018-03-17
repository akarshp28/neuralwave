#!/bin/bash

free -h
sleep 1

echo "enter swap size (format e.g: 15)"
read number

sudo dd if=/dev/zero of=/swapfile bs=1G count=$number
echo "swap size = $number gb"
sleep 1

ls -lh /swapfile
sleep 1

sudo chmod 600 /swapfile
echo "setting permissions"
sleep 1

ls -lh /swapfile
sleep 1

sudo mkswap /swapfile
echo "make swap"
sleep 1

sudo swapon /swapfile
echo "activate swap"
sleep 1

sudo swapon --show
sleep 1

free -h
sleep 1

echo "changes will be lost if pc reboots"
