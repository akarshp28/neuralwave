#include "CSI.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <complex>
#include <vector>
#include <chrono>
#include <thread>
#include "../gnuplot-iostream/gnuplot-iostream.h"
#include <boost/tuple/tuple.hpp>

using namespace std;

int get_len(ifstream &fs);
//void write_csi(vector<CSI> csi_data, string filename);

int main()
{
	char *buffer = new char [3];
	vector<CSI> csi_data;
	char *bytes = NULL;
	uint16_t field_len;
	uint8_t code;
	Gnuplot gp;
	
	//open file
	ifstream fs("log20_sec.dat");

	// get length of file and time of recording:
    int len = get_len(fs);
    auto start = std::chrono::high_resolution_clock::now();

    int cur = 0;

    while (fs.is_open())
    {
    	if (cur+3 < len)
    	{
    		fs.read (buffer, 3);
			field_len = uint8_t(buffer[1]) + (uint8_t(buffer[0]) << 8);		
			code = buffer[2];
			cur += 3;

			if (code == 187)
			{
				if (bytes != NULL)
				{
					delete [] bytes;
				}
				bytes = new char[field_len-1];

				if (cur+field_len-1 <= len)
				{
					fs.read(bytes, field_len-1);
					csi_data.push_back(CSI(bytes));	
					cur += field_len-1;
					
					gp << "set xrange [0:270]\nset yrange [-10:90]\n";
					gp << "plot" << gp.file1d(csi_data[csi_data.size()-1].get_amplitude()) << "with lines " << std::endl;
				}
				else
				{
					fs.seekg(cur-3, fs.beg);
				}
			}
			else
			{
				if (cur+field_len-1 <= len)
				{
					fs.seekg(field_len-1, fs.cur); 
					cur += field_len-1;
				}
				else
				{
					fs.seekg(cur-3, fs.beg);
				}				
			}
    	}
    	else
    	{
    		if ((get_len(fs) - len) != 0)
    		{
    			len = get_len(fs);
    		}
    		else
    		{
    			auto end = std::chrono::high_resolution_clock::now();
    			std::chrono::duration<double> elapsed = end - start;
    			if (elapsed.count() > 10)
    			{
    				break;
    			}
    		}
      	}
    }
    cout << csi_data.size() << endl;
	fs.close();

	return 0;
}

/*
void write_csi(vector<CSI> csi_data, string filename)
{
	ofstream fs(filename);
	for(int i = 0; i < int(csi_data.size()); i++)
	{
		vector<double> amp = csi_data[i].get_amplitude(); 
		vector<double> ph = csi_data[i].get_phase();
		for (int j = 0; j < amp.size(); j++)
		{
			fs << amp[j] << ",";
		}
		for (int j = 0; j < ph.size(); j++)
		{
			fs << ph[j] << ",";
		}
		fs << "/n";
	}
	fs.close();
}
*/

int get_len(ifstream &fs)
{
	int cur = fs.tellg();
	fs.seekg (0, fs.end);
	int length_cur = fs.tellg();
	fs.seekg (cur, fs.beg);

	return length_cur;
}