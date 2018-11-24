/*
 * =====================================================================================
 *       Filename:  sendData.c 
 *
 *    Description:  send packets 
 *        Version:  1.0
 *
 *         Author:  Yaxiong Xie 
 *         Email :  <xieyaxiongfly@gmail.com>
 *   Organization:  WANS group @ Nanyang Technological University 
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 * =====================================================================================
 */
 
#include <arpa/inet.h>
#include <linux/if_packet.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <net/if.h>
#include <netinet/ether.h>
#include <netinet/in.h> 
#include <unistd.h>
#include <errno.h>
#include <time.h> 
#include <uci.h>

/* Define the defult destination MAC address */
#define DEFAULT_DEST_MAC0	0x00
#define DEFAULT_DEST_MAC1	0x03
#define DEFAULT_DEST_MAC2	0x7F
#define DEFAULT_DEST_MAC3	0xB0
#define DEFAULT_DEST_MAC4	0x20
#define DEFAULT_DEST_MAC5	0x20
 
#define IF_5GHz	        	"wlan1"
#define IF_24GHz	    	"wlan0"

#define BUF_SIZ	            2048	
#define PORT 5005 

int setWirelessCfgValue(char *name, char *value, int device)
{
	if (NULL == name || NULL == value)
    {
        return 0;
    }

	char section[20] = {0};
	int ret = 0;
	struct uci_ptr ptr;
	memset(&ptr, 0, sizeof(ptr));
    struct uci_context * ctx = uci_alloc_context();

	if (NULL == ctx)
		printf("setWirelessCfgValue uci_alloc_context error\n");
	
    snprintf(section, sizeof(section), "@wifi-device[%i]", device);
	ptr.package = "wireless",  
	ptr.section = section,  
	ptr.option = name,  
	ptr.value = value,  
	
	ret = uci_set(ctx,&ptr);
	if (ret == 0){
		ret = uci_commit(ctx, &ptr.p, false);
	}
	uci_unload(ctx,ptr.p);
	uci_free_context(ctx);
	
    return 1;
}

int main(int argc, char *argv[])
{
    struct sockaddr_in address;
    int sockfd, server_fd, new_socket, device, Delay, Cnt;
    unsigned int DstAddr[6];
    int addrlen = sizeof(address); 
    char SetupCMD[1024] = {0}; 

    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
       
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons(PORT); 
       
    // Forcefully attaching socket to the port 8080 
    if (bind(server_fd, (struct sockaddr *)&address,  
                                 sizeof(address))<0) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 

    while (true)
    {
        printf("listening!!\n");
	    if (listen(server_fd, 3) < 0) 
	    { 
	        perror("listen"); 
	        exit(EXIT_FAILURE); 
	    } 
	    if ((new_socket = accept(server_fd, (struct sockaddr *)&address,  
	                       (socklen_t*)&addrlen))<0) 
	    { 
	        perror("accept"); 
	        exit(EXIT_FAILURE); 
	    } 

        char buffer[1024] = {0}; 
        int valread = read(new_socket , buffer, 1024); 
        printf("%s %i\n", buffer, (int) time(NULL)); 

        int StartTime;
        char *Command, *Band5, *ChannelWidth, *GI, *Channel, *Power, *Modulation, *MAC, *NumOfPacketToSend, *DelayStr, *TmpTime;
        Command = strtok (buffer," ");
        Band5 = strtok (NULL, " ");
        ChannelWidth = strtok (NULL, " ");
        GI = strtok (NULL, " ");
        Channel = strtok (NULL, " ");
        Power = strtok (NULL, " ");
        Modulation = strtok (NULL, " ");
        MAC = strtok (NULL, " ");
        NumOfPacketToSend = strtok (NULL, " ");
        DelayStr = strtok (NULL, " ");
        TmpTime = strtok (NULL, " ");
        sscanf(TmpTime, "%d", &StartTime);

        char ifName[IFNAMSIZ];
        if (strcmp(Command, "send") == 0)
        {
	        if (strcmp(Band5,"1") == 0)
	        {
	        	strcpy(ifName, IF_5GHz);
	        	device = 1;
	        	setWirelessCfgValue("disabled", "1", 0);
	        	setWirelessCfgValue("disabled", "0", 1);
	        }
	        else
	        {
	        	strcpy(ifName, IF_24GHz);
	        	device = 0;
	        	setWirelessCfgValue("disabled", "0", 0);
	        	setWirelessCfgValue("disabled", "1", 1);
	        }

			setWirelessCfgValue("htmode", ChannelWidth, device);
			setWirelessCfgValue("short_gi_20", GI, device);
			setWirelessCfgValue("short_gi_40", GI, device);
			setWirelessCfgValue("channel", Channel, device);
			setWirelessCfgValue("txpower", Power, device);

			snprintf(SetupCMD, sizeof(SetupCMD), "echo %s >> /sys/kernel/debug/ieee80211/phy%i/rc/fixed_rate_idx", Modulation, device);
			if (system(SetupCMD) != 0) {
			    perror("set MCS rate");
			}

		    sscanf(MAC,"%x:%x:%x:%x:%x:%x",&DstAddr[0],&DstAddr[1],&DstAddr[2],&DstAddr[3],&DstAddr[4],&DstAddr[5]);

		    Cnt = atoi(NumOfPacketToSend);
		    Delay = atoi(DelayStr);
			
			/* Open RAW socket to send on */
			if ((sockfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))) == -1) {
			    perror("socket");
			}
		 
			/* Get the index of the interface to send on */
            struct  ifreq if_idx;
			memset(&if_idx, 0, sizeof(struct ifreq));
			strncpy(if_idx.ifr_name, ifName, IFNAMSIZ-1);
			if (ioctl(sockfd, SIOCGIFINDEX, &if_idx) < 0)
			    perror("SIOCGIFINDEX");

			/* Get the MAC address of the interface to send on */
            struct  ifreq if_mac;
			memset(&if_mac, 0, sizeof(struct ifreq));
			strncpy(if_mac.ifr_name, ifName, IFNAMSIZ-1);
			if (ioctl(sockfd, SIOCGIFHWADDR, &if_mac) < 0)
			    perror("SIOCGIFHWADDR");
		 
			/* Construct the Ethernet header */
            char    sendbuf[BUF_SIZ];
            struct  ether_header *eh = (struct ether_header *) sendbuf;
            struct  iphdr *iph = (struct iphdr *) (sendbuf + sizeof(struct ether_header));
			memset(sendbuf, 0, BUF_SIZ);
			/* Ethernet header */
			eh->ether_shost[0] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[0];
			eh->ether_shost[1] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[1];
			eh->ether_shost[2] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[2];
			eh->ether_shost[3] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[3];
			eh->ether_shost[4] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[4];
			eh->ether_shost[5] = ((uint8_t *)&if_mac.ifr_hwaddr.sa_data)[5];
			eh->ether_dhost[0] = DstAddr[0];
			eh->ether_dhost[1] = DstAddr[1];
			eh->ether_dhost[2] = DstAddr[2];
			eh->ether_dhost[3] = DstAddr[3];
			eh->ether_dhost[4] = DstAddr[4];
			eh->ether_dhost[5] = DstAddr[5];

		    /* Ethertype field */
            int tx_len = 0;
			eh->ether_type = htons(ETH_P_IP);
			tx_len += sizeof(struct ether_header);
		 
			/* Packet data 
		     * We just set it to 0xaa you send arbitrary payload you like*/
		    for(int i=1;i<=1000;i++){
		        
			    sendbuf[tx_len++] = 0xaa;
		    } 
			
            struct  sockaddr_ll socket_address;
		    /* Index of the network device */
			socket_address.sll_ifindex = if_idx.ifr_ifindex;
		    /* RAW communication*/
		    socket_address.sll_family   = PF_PACKET;    
		    /* we don't use a protocoll above ethernet layer
		     *   ->just use anything here*/
		    socket_address.sll_protocol = htons(ETH_P_IP);  
		    
		    /* ARP hardware identifier is ethernet*/
		    socket_address.sll_hatype   = ARPHRD_ETHER;
		        
		    /* target is another host*/
		    socket_address.sll_pkttype  = PACKET_OTHERHOST;
		    
		    /* address length*/
		    socket_address.sll_halen    = ETH_ALEN;
			/* Destination MAC */
			socket_address.sll_addr[0] = DstAddr[0];
			socket_address.sll_addr[1] = DstAddr[1];
			socket_address.sll_addr[2] = DstAddr[2];
			socket_address.sll_addr[3] = DstAddr[3];
			socket_address.sll_addr[4] = DstAddr[4];
			socket_address.sll_addr[5] = DstAddr[5];
		 
		 	while((int) time(NULL) < StartTime)
		 	{
		       	if (usleep(1) == -1){
		            printf("sleep failed\n");
		        }
		 	}

			/* Send packet */
		    for(;Cnt>0;Cnt--)
		    {
		        if (sendto(sockfd, sendbuf, tx_len, 0, (struct sockaddr*)&socket_address, sizeof(struct sockaddr_ll)) < 0){
		            printf("Send failed\n");
		        }

		       	if (usleep(Delay) == -1){
		            printf("sleep failed\n");
		        }
		    }
		}
	}
	
	return 0;
}
