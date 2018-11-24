/*
 * =====================================================================================
 *       Filename:  main.c
 *
 *    Description:  Here is an example for receiving CSI matrix 
 *                  Basic CSi procesing fucntion is also implemented and called
 *                  Check csi_fun.c for detail of the processing function
 *        Version:  1.0
 *
 *         Author:  Yaxiong Xie
 *         Email :  <xieyaxiongfly@gmail.com>
 *   Organization:  WANDS group @ Nanyang Technological University
 *   
 *   Copyright (c)  WANDS group @ Nanyang Technological University
 * =====================================================================================
 */
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <termios.h>
#include <pthread.h>
#include <signal.h>
#include <netinet/in.h> 
#include <sys/socket.h> 
#include <sys/types.h>
#include <sys/stat.h>
#include <time.h> 

#include "csi_fun.h"

#define PORT 5005 
#define BUFSIZE 4096
#define SEND 50

unsigned char buf_addr[BUFSIZE];
unsigned char data_buf[1500];

COMPLEX csi_matrix[3][3][114];
csi_struct*   csi_status;

int main(int argc, char* argv[])
{
    int         verbose = 1, cnt;
    u_int16_t   buf_len;

    int server_fd, new_socket; 
    struct sockaddr_in address; 
    int addrlen = sizeof(address); 
        
    csi_status = (csi_struct*)malloc(sizeof(csi_struct));
    int fd = open_csi_device();

    // Creating socket file descriptor 
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    { 
        perror("socket failed"); 
        exit(EXIT_FAILURE); 
    } 
       
    address.sin_family = AF_INET; 
    address.sin_addr.s_addr = INADDR_ANY; 
    address.sin_port = htons(PORT); 
       
    // Forcefully attaching socket to the port 
    if (bind(server_fd, (struct sockaddr *)&address, sizeof(address))<0) 
    { 
        perror("bind failed"); 
        exit(EXIT_FAILURE); 
    } 

    while(true)
    {
	printf("listening!\n");
        if (listen(server_fd, 3) < 0) 
        { 
            perror("listen"); 
            exit(EXIT_FAILURE); 
        } 
        if ((new_socket = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen))<0) 
        { 
            perror("accept"); 
            exit(EXIT_FAILURE); 
        }

        char buffer[1024] = {0}; 
        int valread = read( new_socket , buffer, 1024); 
	printf("%s %i\n", buffer, (int) time(NULL)); 

        char *Command, *TmpNumberOfSeconds, *TmpTime, *Filename;
        int NumberOfSeconds, StartTime;
        Command = strtok (buffer," ");
        Filename = strtok (NULL, " ");
        TmpNumberOfSeconds = strtok (NULL, " ");
        sscanf(TmpNumberOfSeconds, "%d", &NumberOfSeconds);
        TmpTime = strtok (NULL, " ");
        sscanf(TmpTime, "%d", &StartTime);

        if (strcmp(Command,"recv") == 0)
        {
            FILE* fp = fopen(Filename,"w");
            if (!fp){
                printf("Fail to open <output_file>, are you root?\n");
                fclose(fp);
                return 0;
            }   
          
            if (fd < 0){
                perror("Failed to open the device...");
                return errno;
            }
            
            int total_msg_cnt = 0;
            while(((int) time(NULL) - StartTime) <= NumberOfSeconds+1){
                /* keep listening to the kernel and waiting for the csi report */
                cnt = read_csi_buf(buf_addr,fd,BUFSIZE);

                if (cnt){
                    total_msg_cnt += 1;

                    /* fill the status struct with information about the rx packet */
                    record_status(buf_addr, cnt, csi_status);

                    /* 
                     * fill the payload buffer with the payload
                     * fill the CSI matrix with the extracted CSI value
                     */
                    record_csi_payload(buf_addr, csi_status, data_buf, csi_matrix); 
                                
                    //if (verbose)
                    //{
                    //    printf("Recv %dth msg with rate: 0x%02x | payload len: %d\n",total_msg_cnt,csi_status->rate,csi_status->payload_len);
                    //}

                    /* log the received data for off-line processing */
                    buf_len = csi_status->buf_len;
                    //fwrite(&buf_len,1,2,fp);
                    //fwrite(buf_addr,1,buf_len,fp);

                    if (total_msg_cnt % SEND == 0)
                    {
                        send(new_socket , &buf_len , 2 , 0 ); 
                        send(new_socket , buf_addr , buf_len , 0 ); 
                    }
                }
            }
            fclose(fp);
        }
    }

    close_csi_device(fd);
    free(csi_status);
    return 0;
}
