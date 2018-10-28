/*
Author: Kalvik Jakkala <kjakkala@uncc.edu>
Reference: Based on Daniel Halperin's <dhalperi@cs.washington.edu> intel csi tool

Class to calculate and store csi values from raw byte stream recieved from the intel csi tool
Original Intel CSI tool's website https://dhalperi.github.io/linux-80211n-csitool/index.html
*/

#ifndef __CSI_HPP__
#define __CSI_HPP__

#include <complex>
#include <iostream>
#include <vector>
#include <math.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <thread>
#include "../gnuplot-iostream/gnuplot-iostream.h"
#include <boost/tuple/tuple.hpp>

using namespace std;

class CSI {
private:
   /*
   csi is the CSI itself, normalized to an internal reference. It is 
   a Ntx×Nrx×30 3-D matrix where the third dimension is across 30 
   subcarriers in the OFDM channel. For a 20 MHz-wide channel, 
   these correspond to about half the OFDM subcarriers, and for a 
   40 MHz-wide channel, this is about one in every 4 subcarriers. 
   Which subcarriers were measured is defined by the IEEE 802.11n-2009 
   standard (in Table 7-25f on page 50).
   */
   vector<complex<double>> csi;

   /*
   perm tells us how the NIC permuted the signals from the 3 receive 
   antennas into the 3 RF chains that process the measurements. 
   The sample value of [3 2 1] implies that Antenna C was sent to 
   RF Chain A, Antenna B to Chain B, and Antenna A to Chain C. This 
   operation is performed by an antenna selection module in the NIC 
   and generally corresponds to ordering the antennas in decreasing 
   order of RSSI.
   */
   vector<uint8_t> perm;

   /*
   timestamp_low is the low 32 bits of the NIC's 1 MHz clock. 
   It wraps about every 4300 seconds, or 72 minutes. 
   This field was not yet recorded in the sample trace, 
   so all values are arbitrary and always equal 4.
   */
   uint32_t timestamp_low;

   /*
   bfee_count is simply a count of the total number of beamforming
   measurements that have been recorded by the driver and sent to userspace. 
   The netlink channel between the kernel and userspace is lossy, 
   so these can be used to detect measurements that were dropped in this pipe.
   */
   uint16_t bfee_count;

   /*
   rate is the rate at which the packet was sent, in the same format as the 
   rate_n_flags defined above. Note that the antenna bits are omitted, as 
   there is no way for the receiver to know which transmit antennas were used.
   */
   uint16_t rate;

   /*
   rssi_a, rssi_b, and rssi_c correspond to RSSI measured by the receiving NIC
    at the input to each antenna port. This measurement is made during the 
    packet preamble. This value is in dB relative to an internal reference; 
    to get the received signal strength in dBm we must combine it with the 
    Automatic Gain Control (AGC) setting (agc) in dB and also subtract off 
    a magic constant. This process is explained below.
   */
   uint8_t rssi_a;
   uint8_t rssi_b;
   uint8_t rssi_c;

   /*Nrx represents the number of antennas used to receive the packet by this NIC, 
   and Ntx represents the number of space/time streams transmitted. In this case, 
   the sender sent a single-stream packet and the receiver used all 3 antennas 
   to receive it.
   */
   uint8_t Nrx;
   uint8_t Ntx;
   uint8_t agc;
   int8_t noise;

   //Convert from decibels.
   double dbinv(double x);

   //Calculates the Received Signal Strength (RSS) in dBm
   double get_total_rss();

   //Convert to decibels. Units of x must be 'power'
   double db(double x);

   //scale csi values to absolute units, rather than Intel's internal reference level
   void scale_csi();

   //formats complex numbers for printing
   string print_complex(complex<double> num);

public:
   //Default constructors
   CSI();
   //Overloaded constructor, takes byte stream as input
   CSI(char *inBytes);

   //accessor functions
   vector<complex<double>> get_csi();
   vector<std::pair<double, double>> get_amplitude();
   uint32_t get_timestamp_low();
   vector<double> get_phase();
   vector<uint8_t> get_perm();
   uint16_t get_bfee_count(); 
   uint8_t get_rssi_a();
   uint8_t get_rssi_b();
   uint8_t get_rssi_c();
   uint16_t get_rate();
   int8_t get_noise();
   uint8_t get_Nrx();
   uint8_t get_Ntx();
   uint8_t get_agc();

   //overloaded insertion operator function
   friend ostream &operator<<(ostream &output, CSI &obj);
};

#endif
