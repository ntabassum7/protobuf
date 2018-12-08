#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include "caffe.pb.h"
#include <iostream>


using namespace std;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::Message;
ofstream target ("inception_v1_simple.py");
//used to split a string into tokens depending on delimiter
vector<string> split(string str, string token){
    vector<string>result;
    while(str.size()){
        int index = str.find(token);
        if(index!=string::npos){
            result.push_back(str.substr(0,index));
            str = str.substr(index+token.size());
            if(str.size()==0)result.push_back(str);
        }else{
            result.push_back(str);
            str = "";
        }
    }
    return result;
}

//used to maintain code indenting for python
string indent(int loop){
   string space;
//   cout<<"printing space before"<<endl;
//   cout<<space<<loop;
   loop=loop*3;
   for (int i=0;i<loop;i++)
   {
	space=space+" ";
   }
//   cout<<"printing space after"<<endl;
//   cout<<space<<loop;
return space;
}


void general(caffe::LayerParameter lparam, int ind)
{
	string conv_name,end_point,net;
	string space =indent(ind);
	std::vector<std::string> result,conv_dim;	

	end_point = "end_point = \'"+lparam.name()+"\'";
	result=split(lparam.name(),"_");
	conv_name = result[0];
	conv_dim = split(result[2],"x");
    
	if (lparam.has_convolution_param()) 
	{
		int output = lparam.convolution_param().num_output();
		int stride =lparam.convolution_param().stride()[0];
		//need to figure out the first parameter
		if (stride>1)
			net = "net = slim.conv2d(net, "+std::to_string(output)+", "+"["+conv_dim[0]+","+conv_dim[1]+"], stride="+std::to_string(stride)+", scope=end_point)";
		else
			net = "net = slim.conv2d(net, "+std::to_string(output)+", "+"["+conv_dim[0]+","+conv_dim[1]+"], scope=end_point)";
	}
	     

	if (lparam.has_pooling_param()) 
	{
		int stride = lparam.pooling_param().stride();
		net = "net = slim.max_pool2d(net, ["+conv_dim[0]+","+conv_dim[1]+"], stride="+std::to_string(stride)+", scope=end_point)";
	}
	if((lparam.name().find("_bn") != std::string::npos)||(lparam.name().find("_scale") !=  std::string::npos) || (lparam.name().find("_relu") != std::string::npos))
		return;
		
	target<<space<< end_point<<endl;	
	target<<space<<net<<endl;	
	target<<space<< "end_points[end_point] = net" << endl;
	target<<space<<endl<<endl;
     

}

int main() {
	  caffe::NetParameter param;
	  caffe::LayerParameter lparam;
	  const char * filename = "inception_v1.prototxt";
	  int fd = open(filename, O_RDONLY);
	  if (fd == -1)
	  cout << "File not found: " << filename;
	  google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
	  bool success = google::protobuf::TextFormat::Parse(input, &param);
	ofstream myfile ("example.txt");

	int ind=0;
	string space =indent(ind);
	target<<space<<"from __future__ import absolute_import"<<endl;
	target<<space<<"from __future__ import division"<<endl;
	target<<space<<"from __future__ import print_function"<<endl<<endl;

	target<<space<<"import tensorflow as tf"<<endl<<endl;

	target<<space<<"slim = tf.contrib.slim"<<endl<<endl;
	  myfile << "Network Name: " << param.name() << endl;
	  myfile << "Input: " << param.input(0) << endl;
	  for (int j = 0; j < param.input_dim_size(); j++) 
	  {
	    myfile << "Input Dim "<< j << ": " << param.input_dim(j) << endl;
	  }
		//printing the header part of the target code
		target<<space<<"def "+param.name()+"("+param.input(0)+","<<endl;
		space=indent(ind+5);
                target<<space<< "num_classes=1000,"<<endl;
                target<<space<< "is_training=True,"<<endl;
                target<<space<< "reuse=None,"<<endl;
                target<<space<< "scope=\'"+param.name()+"\'):"<<endl<<endl<<endl;
		space=indent(++ind);
		target<<space<< "with tf.variable_scope(scope, \"Model\", reuse=reuse):"<<endl;
		space=indent(++ind);
		target<<space<< "with slim.arg_scope(default_arg_scope(is_training)):"<<endl<<endl;
		space=indent(++ind);
		target<<space<< "end_points = {}"<<endl<<endl;


	  myfile << "Number of Layers (in implementation): " << param.layer_size() << endl << endl;
	  for (int nlayers = 0; nlayers < param.layer_size(); nlayers++) 
	  {

	    lparam = param.layer(nlayers);
/*
	    //cout<<"Parameters for Layer "<< nlayers + 1 << ":" << endl;
	    myfile << endl << "Parameters for Layer "<< nlayers + 1 << ":" << endl;	 

    for (int num_bottom_layers = 0; num_bottom_layers < lparam.bottom_size(); num_bottom_layers++) {
      myfile << "Bottom: " << lparam.bottom(num_bottom_layers) << endl;
	bottom=lparam.bottom(num_bottom_layers);
    }
    for (int num_top_layers = 0; num_top_layers < lparam.top_size(); num_top_layers++) {
      myfile << "Top: " << lparam.top(num_top_layers) << endl;
    }
    for (int i = 0; i < lparam.param_size(); i++) {
      myfile << "LR_MULT: " << lparam.param(i).lr_mult() << endl;
      myfile << "decay_MULT: " << lparam.param(i).decay_mult() << endl;
    }

*/
	if ((lparam.name().find("Mixed") != std::string::npos)==false)
	{
		general(lparam,ind);
/*
	  if (lparam.has_convolution_param()) 
	  {
	      //myfile << "Number of Outputs: " << lparam.convolution_param().num_output() << endl;
		
		if (lparam.name()== "Logits/Conv2d_0c_1x1")
			cout<<"num_classes: "<< lparam.convolution_param().num_output() << endl;

		//cout << "Name: " << lparam.name() << endl;
		    //cout << "Type: " << lparam.type() << endl;
		//cout << "Pad: " << lparam.convolution_param().pad()[0] << endl;
	      //cout << "Kernel Size: " << lparam.convolution_param().kernel_size()[0] << endl;
	      //cout << "Stride: " << lparam.convolution_param().stride()[0] << endl;
	      //myfile << "Group: " << lparam.convolution_param().group() << endl;
	    }
	     

	  if (lparam.has_lrn_param()) {
	      cout << "Local Size: " << lparam.lrn_param().local_size() << endl;
	      cout << "Alpha: " << lparam.lrn_param().alpha() << endl;
	      cout << "Beta: " << lparam.lrn_param().beta() << endl;
	    }
	    if (lparam.has_pooling_param()) {
	      //cout << "Pool: " << lparam.pooling_param().pool() << endl;
	      //cout << "Kernel Size: " << lparam.pooling_param().kernel_size() << endl;
	      //cout << "Stride: " << lparam.pooling_param().stride() << endl;
	    }
	    if (lparam.has_inner_product_param()) {
	      myfile << "Number of Outputs: " << lparam.inner_product_param().num_output() << endl;
	    }
	    if (lparam.has_dropout_param()) {
	      myfile << "Dropout Ratio: " << lparam.dropout_param().dropout_ratio() << endl;
	    }

*/
	   }


  
  else{
	target<<"Parameters for Layer "<< nlayers + 1 << ":" << endl;
	target<< "Name: " << lparam.name() << endl;
	std::vector<std::string> result=split(lparam.name(),"/");
	end_point = "end_point = \'"+result[0]+"\'";
	target<<space<< end_point<<endl;	
	target<<space<< "with tf.variable_scope(end_point):"<<endl;
	space=indent(++ind);
	target<<space<< "with tf.variable_scope(\'"+result[1]+"\'):"<<endl;
	space=indent(++ind);
	
	

	
  }
 } 
  delete input;
  //close(fd);*/
   
   
return 0;
}
