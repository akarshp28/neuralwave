#include "CSI.hpp"

CSI::CSI()
{
   this->timestamp_low = 0;
   this->bfee_count = 0;
   this->rate = 0;
   this->rssi_a = 0;
   this->rssi_b = 0;
   this->rssi_c = 0;
   this->Nrx = 0;
   this->Ntx = 0;
   this->agc = 0;
   this->noise = 0;
}

CSI::CSI(char *inBytes)
{
   //What perm should sum to for 1,2,3 antennas
   uint8_t triangle [3] = {0, 1, 3};

   this->timestamp_low = uint8_t(inBytes[0]) + (uint8_t(inBytes[1]) << 8) + (uint8_t(inBytes[2]) << 16) + (uint8_t(inBytes[3]) << 24);
   this->bfee_count = uint8_t(inBytes[4]) + (uint8_t(inBytes[5]) << 8);
   this->Nrx = inBytes[8]; 
   this->Ntx = inBytes[9];
   this->rssi_a = inBytes[10];
   this->rssi_b = inBytes[11];
   this->rssi_c = inBytes[12];
   this->noise = inBytes[13];
   this->agc = inBytes[14];
   this->rate = uint8_t(inBytes[18]) + (uint8_t(inBytes[19]) << 8);
   this->csi.resize(this->Ntx * this->Nrx * 30);

   unsigned int antenna_sel = inBytes[15];
   this->perm.push_back((antenna_sel) & 0x3);
   this->perm.push_back((antenna_sel >> 2) & 0x3);
   this->perm.push_back((antenna_sel >> 4) & 0x3);

   unsigned int len = inBytes[16] + (uint8_t(inBytes[17]) << 8);
   unsigned int calc_len = (30 * (this->Nrx * this->Ntx * 8 * 2 + 3) + 7) / 8;

   if (len != calc_len)
   {
      throw string("Wrong beamforming matrix size.\n");
   }

   unsigned int index = 0, remainder;
   char *payload = &inBytes[20];
   char tmp_r, tmp_i;
   int l;
   bool broken_perm = false;

   // check if matrix does not contain default values
   if ((this->perm[0] + this->perm[1] + this->perm[2]) != triangle[this->Nrx-1])
   {
      cerr << "WARN ONCE: Found CSI with Nrx=" << unsigned(this->Nrx) << " and invalid perm=[" << unsigned(this->perm[0]) << ", " << unsigned(this->perm[1]) << ", " << unsigned(this->perm[2]) << "]\n";
      broken_perm = true;
   }
   
   for (int k = 0; k < 30; k++)
   {
      index += 3;
      remainder = index % 8;
      for (int j = 0; j < this->Nrx; j++)
      {
         for (int i = 0; i < this->Ntx; i++)
         {
            tmp_r = (uint8_t(payload[index / 8]) >> remainder) | (uint8_t(payload[index/8+1]) << (8-remainder));
            tmp_i = (uint8_t(payload[index / 8+1]) >> remainder) |   (uint8_t(payload[index/8+2]) << (8-remainder));
            //No permuting needed for only 1 antenna
            if(this->Nrx == 1)
            {
               this->csi[i*this->Nrx*30 + j*30 + k] = complex<double> (tmp_r, tmp_i);
            }
            else if (!broken_perm)
            {
               this->csi[i*this->Nrx*30 + this->perm[j]*30 + k] = complex<double> (tmp_r, tmp_i);
            }
            else
            {
               this->csi[i*this->Nrx*30 + j*30 + k] = complex<double> (tmp_r, tmp_i);
            }
            index += 16;
         }
      }
   }

   this->scale_csi();
}

uint32_t CSI::get_timestamp_low()
{
   return this->timestamp_low;
}

uint16_t CSI::get_bfee_count()
{
   return this->bfee_count;
}

uint16_t CSI::get_rate()
{
   return this->rate;
}

uint8_t CSI::get_rssi_a()
{
   return this->rssi_a;
}

uint8_t CSI::get_rssi_b()
{
   return this->rssi_b;
}

uint8_t CSI::get_rssi_c()
{
   return this->rssi_c;
}

uint8_t CSI::get_Nrx()
{
   return this->Nrx;
}

uint8_t CSI::get_Ntx()
{
   return this->Ntx;
}

uint8_t CSI::get_agc()
{
   return this->agc;
}

int8_t CSI::get_noise()
{
   return this->noise;
}

vector<complex<double>> CSI::get_csi()
{
   return this->csi;
}

vector<uint8_t> CSI::get_perm()
{
   return this->perm;
}

void CSI::scale_csi()
{
   double csi_pwr = 0;
   double noise_db;
   for (int i = 0; i < this->Ntx*this->Nrx*30; i++)
   {
      csi_pwr += real(this->csi[i]*conj(this->csi[i]));
   }

   double rssi_pwr = this->dbinv(this->get_total_rss());
   //Scale CSI -> Signal power : rssi_pwr / (mean of csi_pwr)
   double scale = rssi_pwr / (csi_pwr / 30.0);

   // Thermal noise might be undefined if the trace was
   // captured in monitor mode. If so, set it to -92
   if (this->noise == -127)
   {
      noise_db = -92;
   }
   else
   {
      noise_db = this->noise;
   }

   double thermal_noise_pwr = dbinv(noise_db);

   // Quantization error: the coefficients in the matrices are
   // 8-bit signed numbers, max 127/-128 to min 0/1. Given that Intel
   // only uses a 6-bit ADC, I expect every entry to be off by about
   // +/- 1 (total across real & complex parts) per entry.
   //
   // The total power is then 1^2 = 1 per entry, and there are
   // Nrx*Ntx entries per carrier. We only want one carrier's worth of
   // error, since we only computed one carrier's worth of signal above.
   double quant_error_pwr = scale * (this->Nrx * this->Ntx);

   // Total noise and error power
   double total_noise_pwr = thermal_noise_pwr + quant_error_pwr;

   // Ret now has units of sqrt(SNR) just like H in textbooks
   if (this->Ntx == 1)
   {
      for (int i = 0; i < this->Ntx*this->Nrx*30; i++)
      {
         this->csi[i] *= sqrt(scale / total_noise_pwr);
      }
   }
   else if (this->Ntx == 2)
   {
      for (int i = 0; i < this->Ntx*this->Nrx*30; i++)
      {
         this->csi[i] *= sqrt(2) * sqrt(scale / total_noise_pwr);
      }
   }
   else if (this->Ntx == 3)
   {
      // Note: this should be sqrt(3)~ 4.77 dB. But, 4.5 dB is how
      // Intel (and some other chip makers) approximate a factor of 3
      for (int i = 0; i < this->Ntx*this->Nrx*30; i++)
      {
         this->csi[i] *= sqrt(this->dbinv(4.5)) * sqrt(scale / total_noise_pwr);
      }
   }
}

double CSI::get_total_rss()
{
   double rssi_mag = 0;
   if (this->rssi_a != 0)
   {
      rssi_mag += this->dbinv(this->rssi_a);
   }

   if (this->rssi_b != 0)
   {
      rssi_mag += this->dbinv(this->rssi_b);
   }

   if (this->rssi_c != 0)
   {
      rssi_mag += this->dbinv(this->rssi_c);
   }
   return this->db(rssi_mag) - 44.0 - this->agc;
}

double CSI::dbinv(double x)
{
   return pow(10.0, (x/10.0));
}

double CSI::db(double x)
{
   if (x < 0)
   {
      throw string("CSI::db Input must be positive\n");
   }  
   else
   {
      return (10.0 * log10(x) + 300) - 300;
   }
}

vector<std::pair<double, double>> CSI::get_amplitude()
{
   vector<std::pair<double, double>> tmp(this->Ntx*this->Nrx*30);
   for (int i = 0; i < this->Ntx*this->Nrx*30; i++)
   {
      tmp.push_back(std::make_pair(i, abs(this->csi[i])));
   }

   return tmp;
}

vector<double> CSI::get_phase()
{
   vector<double> tmp(this->Ntx*this->Nrx*30);
   for (int i = 0; i < this->Ntx*this->Nrx*30; i++)
   {
      tmp[i] = arg(this->csi[i]);
   }
   return tmp;
}

ostream &operator<<(ostream &output, CSI &obj) 
{
   output << "timestamp_low : " << obj.get_timestamp_low() << endl; 
   output << "bfee_count    : " << obj.get_bfee_count() << endl; 
   output << "perm          : [" << unsigned(obj.get_perm()[0]) << ", " << unsigned(obj.get_perm()[1]) << ", " << unsigned(obj.get_perm()[2]) << "]" << endl;
   output << "rate          : " << obj.get_rate() << endl; 
   output << "rssi_a        : " << unsigned(obj.get_rssi_a()) << endl; 
   output << "rssi_b        : " << unsigned(obj.get_rssi_b()) << endl; 
   output << "rssi_c        : " << unsigned(obj.get_rssi_c()) << endl; 
   output << "Ntx           : " << unsigned(obj.get_Ntx()) << endl; 
   output << "Nrx           : " << unsigned(obj.get_Nrx()) << endl; 
   output << "agc           : " << unsigned(obj.get_agc()) << endl; 
   output << "noise         : " << signed(obj.get_noise()) << endl; 
   output << "csi           : " << endl;

   vector<complex<double>> csi = obj.get_csi();
   for (int i = 0; i<30; i++)
   {
      output << "\tcolumn " << i << endl;
      for (int j = 0; j < obj.get_Ntx(); j++)
      {
         output << "\t";
         for (int k = 0; k < obj.get_Nrx(); k++)
         {
            output << obj.print_complex(csi[j*obj.get_Nrx()*30 + k*30 + i]) << " ";
         }
         output << "\n";
      }
      output << "\n";
   }
   
   return output;
}

string CSI::print_complex(complex<double> num) 
{
   std::ostringstream ss;
   ss << "[" << std::fixed;
   ss << setw(6) << setfill('0') << std::internal << std::setprecision(2) << real(num) << ", ";
   ss << setw(6) << setfill('0') << std::internal << std::setprecision(2) << imag(num) << "i]";
   return ss.str();
}