#include <iostream>
#include <complex>
using namespace std;

class bfee
{
  public:
    void read_bfee(unsigned char *inBytes);

    unsigned int timestamp_low ;
  	unsigned short bfee_count;
  	unsigned int Nrx;
  	unsigned int Ntx;
  	unsigned int rssi_a;
  	unsigned int rssi_b;
  	unsigned int rssi_c;
  	char noise;
  	unsigned int agc;
    unsigned int perm[3];
  	unsigned int rate;
    complex<double> *csi;
};

bfee::bfee()
{
  timestamp_low = 0;
  bfee_count = 0;
  Nrx = 0;
  Ntx = 0;
  rssi_a = 0;
  rssi_b = 0;
  rssi_c = 0;
  noise = '';
  agc = 0;
  perm[0] = 0;
  perm[1] = 0;
  perm[2] = 0;
  rate = 0;
  csi = NULL;
}

/* The computational routine */
void bfee::read_bfee(unsigned char *inBytes)
{
  timestamp_low = inBytes[0] + (inBytes[1] << 8) + (inBytes[2] << 16) + (inBytes[3] << 24);
	bfee_count = inBytes[4] + (inBytes[5] << 8);
  Nrx = inBytes[8];
	Ntx = inBytes[9];
	rssi_a = inBytes[10];
  rssi_b = inBytes[11];
  rssi_c = inBytes[12];
	noise = inBytes[13];
	agc = inBytes[14];
	unsigned int antenna_sel = inBytes[15];
	unsigned int len = inBytes[16] + (inBytes[17] << 8);
	rate = inBytes[18] + (inBytes[19] << 8);
	unsigned int calc_len = (30 * (Nrx * Ntx * 8 * 2 + 3) + 7) / 8;
  unsigned char *payload = &inBytes[20];
  char real, img;
  unsigned int index = 0, remainder;
  csi = new complex<double> [Ntx*Nrx*30];

	/* Check that length matches what it should */
	if (len != calc_len)
		cout << "Wrong beamforming matrix size.\n";

	/* Compute CSI from all this crap :) */
	for (unsigned int i = 0; i < 30; ++i)
	{
		index += 3;
		remainder = index % 8;
		for (unsigned int j = 0; j < Ntx; ++j)
		{
      for (unsigned int k = 0; k < Nrx; ++k)
      {
        real = (payload[index / 8] >> remainder) |
  				(payload[index/8+1] << (8-remainder));
        img = (payload[index / 8+1] >> remainder) |
    				(payload[index/8+2] << (8-remainder));

  			csi[ (j * Nrx * 30) + (k * 30) + i ] = complex<double>(real, img);
  			index += 16;
      }
		}
	}

	/* Compute the permutation array */
	perm[0] = ((antenna_sel) & 0x3);
	perm[1] = ((antenna_sel >> 2) & 0x3);
	perm[2] = ((antenna_sel >> 4) & 0x3);
}

extern "C" {
    bfee* bfee_c(){ return new bfee(); }
    void read_bfee_c(bfee* Bfee, unsigned char *bytes){ Bfee->read_bfee(bytes); }
    unsigned int get_timestamp_low(bfee* Bfee){ return Bfee->timestamp_low; }
    unsigned short get_bfee_count(bfee* Bfee){ return Bfee->bfee_count; }
    unsigned int get_Nrx(bfee* Bfee){ return Bfee->Nrx; }
    unsigned int get_Ntx(bfee* Bfee){ return Bfee->Ntx; }
    unsigned int get_rssi_a(bfee* Bfee){ return Bfee->rssi_a; }
    unsigned int get_rssi_b(bfee* Bfee){ return Bfee->rssi_b; }
    unsigned int get_rssi_c(bfee* Bfee){ return Bfee->rssi_c; }
    char get_noise(bfee* Bfee){ return Bfee->noise; }
    unsigned int get_agc(bfee* Bfee){ return Bfee->agc; }
    unsigned int* get_perm(bfee* Bfee){ return Bfee->perm; }
    unsigned int get_rate(bfee* Bfee){ return Bfee->rate; }
    complex<double>* get_csi(bfee* Bfee){ return Bfee->csi; }
    void del_obj(bfee* Bfee){ delete [] Bfee->csi; }
}

//   g++ -c -fPIC bfee.cpp -o bfee.o
//   g++ -shared -Wl,-soname,libbfee.so -o libbfee.so  bfee.o
