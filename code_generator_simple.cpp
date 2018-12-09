#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
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

//target file
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
        }
	else{
            result.push_back(str);
            str = "";
        }
    }
    return result;
}

//used to maintain code indenting for python
string indent(int loop){
   string space;
   loop=loop*3;
   for (int i=0;i<loop;i++)
	space=space+" ";
return space;
}

//used to print general layers
void general(caffe::LayerParameter lparam, int ind)
{
	string conv_name,end_point,net, slim;
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
		if (result[0]=="AvgPool")
			slim = "slim.avg_pool2d";
		else
			slim = "slim.max_pool2d";
		net = "net = "+ slim +"(net, ["+conv_dim[0]+","+conv_dim[1]+"], stride="+std::to_string(stride)+", scope=end_point)";
	}
	if((lparam.name().find("_bn") != std::string::npos)||(lparam.name().find("_scale") !=  std::string::npos) || (lparam.name().find("_relu") != std::string::npos))
		return;
		
	target<<space<< end_point<<endl;	
	target<<space<<net<<endl;	
	target<<space<< "end_points[end_point] = net" << endl;
	target<<space<<endl<<endl;
}

//used to print mixed layers
void mixed(caffe::LayerParameter lparam, int ind, string bottom)
{
	if((lparam.name().find("_bn") != std::string::npos)||(lparam.name().find("_scale") !=  std::string::npos) || (lparam.name().find("_relu") != std::string::npos))
		return;
	string end_point, net;
	string space =indent(ind);
	std::vector<std::string> result, conv_dim, inter;
	result=split(lparam.name(),"/");
	inter = split(result[2],"_");
	//conv_name = inter[0];
	conv_dim = split(inter[2],"x");

	if (lparam.has_convolution_param())
	{
		int output = lparam.convolution_param().num_output();
		int stride =lparam.convolution_param().stride()[0];
		if (stride>1)		
			net = result[1]+" = slim.conv2d("+bottom+", "+std::to_string(output)+", "+"["+conv_dim[0]+","+conv_dim[1]+"], stride="+std::to_string(stride)+", scope="+result[2]+")";
		else
			net = result[1]+" = slim.conv2d("+bottom+", "+std::to_string(output)+", "+"["+conv_dim[0]+","+conv_dim[1]+"], scope="+result[2]+")";
	}
	if (lparam.has_pooling_param())
	{
		int stride =lparam.pooling_param().stride();
		if (stride>1)		
			net = result[1]+" = slim.max_pool2d("+bottom+", ["+conv_dim[0]+","+conv_dim[1]+"], stride="+std::to_string(stride)+", scope="+result[2]+")";
		else
			net = result[1]+" = slim.max_pool2d("+bottom+", ["+conv_dim[0]+","+conv_dim[1]+"], scope="+result[2]+")";
	} 
	target<<space<<net<<endl;	

}

//used to print the code in the last part
void end_code()
{
int ind = 0;
string space=indent(ind);

target<<space<< "def default_arg_scope(is_training=True, "<<endl;
space=indent(ind+5);
target<<space<< "weight_decay=0.00004,"<<endl;
target<<space<< "use_batch_norm=True,"<<endl;
target<<space<< "batch_norm_decay=0.9997,"<<endl;
target<<space<< "batch_norm_epsilon=0.001,"<<endl;
target<<space<< "batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):"<<endl;

space=indent(++ind);
target<<space<< "batch_norm_params = {"<<endl;

space=indent(ind+1);
target<<space<< "\'decay\': batch_norm_decay,"<<endl;
target<<space<< "\'epsilon\': batch_norm_epsilon,"<<endl;
target<<space<< "\'updates_collections\': batch_norm_updates_collections,"<<endl;
target<<space<< "\'fused\': None,"<<endl;

space=indent(ind);
target<<space<< "}"<<endl;

target<<space<< "if use_batch_norm:"<<endl;
space=indent(ind+1);
target<<space<< "normalizer_fn = slim.batch_norm"<<endl;
target<<space<< "normalizer_params = batch_norm_params"<<endl;
space=indent(ind);
target<<space<< "else:"<<endl;
space=indent(ind+1);
target<<space<< "normalizer_fn = None"<<endl;
target<<space<< "normalizer_params = {}"<<endl<<endl;

space=indent(ind);
target<<space<< "with slim.arg_scope([slim.batch_norm, slim.dropout],"<<endl;
space=indent(ind+5);
target<<space<< "is_training=is_training):"<<endl;
space=indent(++ind);
target<<space<< "with slim.arg_scope([slim.conv2d, slim.fully_connected],"<<endl;
space=indent(ind+5);                        
target<<space<< "weights_regularizer=slim.l2_regularizer(weight_decay)):"<<endl;
space=indent(++ind);
target<<space<< "with slim.arg_scope("<<endl;
space=indent(++ind); 
target<<space<< "[slim.conv2d],"<<endl;
target<<space<< "normalizer_fn=normalizer_fn,"<<endl;
target<<space<< "normalizer_params=normalizer_params):"<<endl;
space=indent(++ind); 
target<<space<< "with slim.arg_scope([slim.conv2d, slim.max_pool2d],"<<endl;
space=indent(ind+2); 
target<<space<< "stride=1, padding='SAME') as sc:"<<endl;
space=indent(++ind); 
target<<space<< "return sc"<<endl;

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

	int ind=0, axis, j;
	std::string previous_endpoint, previous_branch, values, logit_endpoint;
	string space =indent(ind);
	target<<space<<"from __future__ import absolute_import"<<endl;
	target<<space<<"from __future__ import division"<<endl;
	target<<space<<"from __future__ import print_function"<<endl<<endl;

	target<<space<<"import tensorflow as tf"<<endl<<endl;

	target<<space<<"slim = tf.contrib.slim"<<endl<<endl;
	  cout << "Network Name: " << param.name() << endl;
	  cout << "Input: " << param.input(0) << endl;
//	cout<< "Input dim size: "<<param.input_dim_size()<<endl;
//	cout<< "Input shape: "<<param.input_shape()<<endl;

	  for (j = 0; j < param.input_dim_size(); j++); 
	  {
//	    cout << "Input Dim "<< j << ": " << param.input_dim(j) << endl;
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

		if ((lparam.name().find("Mixed") != std::string::npos))
		{
			//cout << "Name: " << lparam.name() << endl;
			string end_point, bottom;
			std::vector<std::string> result;
			if (lparam.type()=="Concat")
			{
				values = values + "])";
				target<<space<< "net = tf.concat("<<endl;
				space=indent(ind+1);
				target<<space<<"axis="+std::to_string(axis-1)+", "+values<<endl;
				space=indent(--ind);
				target<<space<< "end_points[end_point] = net"<<endl<<endl<<endl;
			}
			else
			{
				result=split(lparam.name(),"/");
				bottom=result[1];
				if (previous_endpoint.empty() || previous_endpoint != result[0])
				{		
					axis= 0;
					values = "values=["+ result[1];
					end_point = "end_point = \'"+result[0]+"\'";	
					target<<space<< end_point<<endl;	
					target<<space<< "with tf.variable_scope(end_point):"<<endl;
					space=indent(++ind);
				}
				if (previous_branch.empty() || previous_branch != result[1])
				{
					target<<space<< "with tf.variable_scope(\'"+result[1]+"\'):"<<endl;
					//space=indent(ind+1);
					bottom="net";
					if (axis>0)
						values = values + ", " + result[1];
					axis++;
				}
				mixed(lparam,ind+1,bottom);
				previous_endpoint = result[0];
				previous_branch = result[1];
			}	
	  	}
		else if ((lparam.name().find("Logits") != std::string::npos))
		{
			std::vector<std::string> result, inter, conv_dim;
			result=split(lparam.name(),"/");
			
//			bottom=result[1];
			if (logit_endpoint.empty())
			{
				target<<space<< "end_point = 'Logits'"<<endl;
				target<<space<< "with tf.variable_scope(end_point):"<<endl;
				space=indent(++ind);
				logit_endpoint = "flag";
			}
			if (lparam.has_dropout_param())
			{
//				std::setprecision(1);
//				target  << endl;
				stringstream stream;
				stream << fixed << setprecision(1) << 1-lparam.dropout_param().dropout_ratio();
				string s = stream.str();
				target<<space<< std::fixed << std::setprecision(1)<< "net = slim.dropout(net, "+s+", scope=\'"+result[1]+"\')"<<endl;
			}
			if (lparam.has_convolution_param())
			{
				inter = split(result[1],"_");
				conv_dim = split(inter[2],"x");
			        target<<space<< "logits = slim.conv2d(net, num_classes, ["+conv_dim[0]+","+conv_dim[1]+"], activation_fn=None,"<<endl;
				space=indent(ind+7);
				target<<space<< "normalizer_fn=None, scope=\'"+result[1]+"\')"<<endl;
			}
		}
		else if (lparam.name()=="Predictions")
		{
			space=indent(ind);
			target<<space<< "logits = tf.squeeze(logits, [1, 2], name=\'SpatialSqueeze\')"<<endl;
        		target<<space<< "end_points[end_point] = logits"<<endl;
			space=indent(--ind);
			target<<space<< "end_points[\'Predictions\'] = slim."+lparam.type()+"(logits, scope=\'Predictions\')"<<endl;
			space=indent(--ind);
			target<<space<< "return logits, end_points"<<endl;
			space=indent(--ind);
			//target<<space<< param.name()<<".default_image_size = "<<param.input_dim(2)<<endl;
			end_code();
		
		}
		else 
		{
			general(lparam,ind);

		}
 
 	} 
  delete input;
  //close(fd);*/
   
   
return 0;
}
