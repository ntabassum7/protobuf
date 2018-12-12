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
ofstream target ("inception_v1_multiplexing.py");

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

int find_class(caffe::NetParameter param)
{
	caffe::LayerParameter lparam;
	for (int nlayers=0; nlayers < param.layer_size(); nlayers++) 
	{
		lparam = param.layer(nlayers);
		if(lparam.type()=="Convolution" && lparam.bottom()[0]==lparam.top()[0] && lparam.bottom()[0]=="logits")
			return lparam.convolution_param().num_output();
	}
	return 1000;
		
}

//used to print general layers
void general(caffe::LayerParameter lparam, int ind, string ep)
{
	string conv_name,end_point,net, slim, conv_dim;
	string space =indent(ind);

	end_point = "end_point = \'"+lparam.name()+"\'";

	if (lparam.has_convolution_param()) 
	{
		int output = lparam.convolution_param().num_output();
		int stride =lparam.convolution_param().stride()[0];
		conv_dim = std::to_string(lparam.convolution_param().kernel_size()[0]);

		if (stride>1)
			net = "net = slim.conv2d("+ep+", "+std::to_string(output)+", "+"["+conv_dim+","+conv_dim+"], stride="+std::to_string(stride)+", scope=end_point)";
		else
			net = "net = slim.conv2d("+ep+", "+std::to_string(output)+", "+"["+conv_dim+","+conv_dim+"], scope=end_point)";
	}
	     

	if (lparam.has_pooling_param()) 
	{
		int stride = lparam.pooling_param().stride();
		conv_dim = std::to_string(lparam.pooling_param().kernel_size());

		if(lparam.pooling_param().has_global_pooling())
			slim = "slim.avg_pool2d";		
		else
			slim = "slim.max_pool2d";
		net = "net = "+ slim +"("+ep+", ["+conv_dim+","+conv_dim+"], stride="+std::to_string(stride)+", scope=end_point)";
	}
		
	target<<space<< end_point<<endl;	
	target<<space<<net<<endl;	
	target<<space<< "end_points[end_point] = net" << endl;
	target<<space<<endl<<endl;
}

//used to print mixed layers
void mixed(caffe::LayerParameter lparam, int ind, string ep, string b_num, string select, string close)
{
	string end_point, net, conv_dim, name=lparam.name();
	string space =indent(ind);
	std::vector<std::string> result;
	if (lparam.name().find("/")!=string::npos)
	{
		result=split(lparam.name(),"/");
		name = result[2];	
	}


	if (lparam.has_convolution_param())
	{
		int output = lparam.convolution_param().num_output();
		int stride =lparam.convolution_param().stride()[0];
		conv_dim = std::to_string(lparam.convolution_param().kernel_size()[0]);
		if (stride>1)		
			net = b_num+" = slim.conv2d("+ep+", "+select+std::to_string(output)+close+", "+"["+conv_dim+","+conv_dim+"], stride="+std::to_string(stride)+", scope=\'"+name+"\')";
		else
			net = b_num+" = slim.conv2d("+ep+", "+select+std::to_string(output)+close+", "+"["+conv_dim+","+conv_dim+"], scope=\'"+name+"\')";
	}
	if (lparam.has_pooling_param())
	{
		conv_dim = std::to_string(lparam.pooling_param().kernel_size());
		int stride =lparam.pooling_param().stride();
		if (stride>1)		
			net = b_num+" = slim.max_pool2d("+ep+", ["+conv_dim+","+conv_dim+"], stride="+std::to_string(stride)+", scope=\'"+name+"\')";
		else
			net = b_num+" = slim.max_pool2d("+ep+", ["+conv_dim+","+conv_dim+"], scope=\'"+name+"\')";
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
target<<space<< "batch_norm_updates_collections=tf.GraphKeys.UPDATE_OPS):"<<endl<<endl;

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

int check_branch(caffe::NetParameter param, int nlayers, string name)
{
	caffe::LayerParameter lparam;
	int count = 0, flag=0;
	for (; nlayers < param.layer_size(); nlayers++) 
	{
		lparam = param.layer(nlayers);

		if((lparam.type() == "BatchNorm")||(lparam.type() == "Scale") || (lparam.type() == "ReLU"))
			continue;
		if (name==lparam.bottom()[0])
			count++;
	}

	if (count>1)
		flag=1;
	return flag;
}

void dropout(caffe::LayerParameter lparam, int ind)
{
	std::vector<std::string> result;
	string conv_dim, space, name=lparam.name();
	
	if (lparam.name().find("/")!=string::npos)
	{
		result=split(lparam.name(),"/");
		name = result[1];	
	}
	
	if (lparam.has_dropout_param())
	{
		space=indent(ind);	
		target<<space<< "end_point = 'Logits'"<<endl;
		target<<space<< "with tf.variable_scope(end_point):"<<endl;
		space=indent(++ind);
		stringstream stream;
		stream << fixed << setprecision(1) << 1-lparam.dropout_param().dropout_ratio();
		string s = stream.str();
		target<<space<< std::fixed << std::setprecision(1)<< "net = slim.dropout(net, "+s+", scope=\'"+name+"\')"<<endl;
	}
	if (lparam.has_convolution_param())
	{
		space=indent(++ind);	
		conv_dim = std::to_string(lparam.convolution_param().kernel_size()[0]);
	        target<<space<< "logits = slim.conv2d(net, num_classes, ["+conv_dim+","+conv_dim+"], activation_fn=None,"<<endl;
		space=indent(ind+7);
		target<<space<< "normalizer_fn=None, scope=\'"+name+"\')"<<endl;
		space=indent(ind);
		target<<space<< "logits = tf.squeeze(logits, [1, 2], name=\'SpatialSqueeze\')"<<endl;
		target<<space<< "end_points[end_point] = logits"<<endl;
	}
}

string find_concat(caffe::NetParameter param, int nlayers, string name)
{
	caffe::LayerParameter lparam;
	int flag=0;
	for (; nlayers < param.layer_size(); nlayers++) 
	{
		lparam = param.layer(nlayers);

		if(lparam.type() != "Concat")
			continue;
		for(int i=0;i<lparam.bottom_size();i++)
			if(name==lparam.bottom()[i])
				return lparam.name();
	}
	return name;
}

int main() {
	  caffe::NetParameter param;
	  caffe::LayerParameter lparam, oldparam;
	  const char * filename = "inception_v1.prototxt";
	  int fd = open(filename, O_RDONLY);
	  if (fd == -1)
	  cout << "File not found: " << filename;
	  google::protobuf::io::FileInputStream* input = new google::protobuf::io::FileInputStream(fd);
	  bool success = google::protobuf::TextFormat::Parse(input, &param);

	int ind=0, axis=0, j, branch=0, num_classes;
	std::string oldbottom, previous_endpoint, previous_branch, values, gen_ep, br_concat;
	string space =indent(ind);
	target<<space<<"from __future__ import absolute_import"<<endl;
	target<<space<<"from __future__ import division"<<endl;
	target<<space<<"from __future__ import print_function"<<endl<<endl;
	target<<space<<"import tensorflow as tf"<<endl<<endl;
	target<<space<<"slim = tf.contrib.slim"<<endl<<endl;
	
	gen_ep=param.input(0);
	num_classes = find_class(param);

	//printing the header part of the target code
	target<<space<<"def "+param.name()+"("+param.input(0)+","<<endl;
	space=indent(ind+5);
        target<<space<< "num_classes="<<std::to_string(num_classes)<<","<<endl;
        target<<space<< "is_training=True,"<<endl;
        target<<space<< "reuse=None,"<<endl;
        target<<space<< "scope=\'"+param.name()+"\'):"<<endl;
	//changed for multiplexing code
	target<<space<< "config=None):"<<endl<<endl<<endl;
	space=indent(++ind);

	//changed for multiplexing code
	target<<space<< "selectdepth = lambda k,v: int(config[k][\'ratio\']*v) if config and k in config and \'ratio\' in config[k] else v"<<endl<<endl; 

	//changed for multiplexing code
	target<<space<< "selectinput = lambda k, v: config[k][\'input\'] if config and k in config and \'input\' in config[k] else v"<<endl<<endl; 

	

	target<<space<< "with tf.variable_scope(scope, \"Model\", reuse=reuse):"<<endl;
	space=indent(++ind);
	target<<space<< "with slim.arg_scope(default_arg_scope(is_training)):"<<endl<<endl;
	space=indent(++ind);
	target<<space<< "end_points = {}"<<endl<<endl;

	for (int nlayers = 0; nlayers < param.layer_size(); nlayers++) 
	{

		lparam = param.layer(nlayers);
		if((lparam.type() == "BatchNorm")||(lparam.type() == "Scale") || (lparam.type() == "ReLU"))
			continue;
		else if (lparam.type()== "Dropout")
		{
			branch = 2;
			dropout(lparam,ind);
		}
		else if (lparam.type()=="Softmax")
		{
			target<<space<< "end_points[\'Predictions\'] = slim."+lparam.type()+"(logits, scope=\'Predictions\')"<<endl;
			space=indent(--ind);
			target<<space<< "return logits, end_points"<<endl<<endl;
			space=indent(--ind);	
			target<<space<< param.name()<<".default_image_size = "<<param.input_shape()[0].dim()[2]<<endl<<endl<<endl;
			end_code();
		}
		else
		{
			if (branch==0)
			{
				general(lparam,ind,gen_ep);
				gen_ep="net";
				branch=check_branch(param, nlayers, lparam.name());
			}

			else if (branch==1)
			{
				string end_point, ep_send, current_branch, b_num, select, close;
				if (lparam.type()=="Concat")
				{
					br_concat="";
					values = values + "])";
					target<<space<< "net = tf.concat("<<endl;
					space=indent(ind+1);
					target<<space<<"axis="+std::to_string(axis-1)+", "+values<<endl;
					space=indent(--ind);
					target<<space<< "end_points[end_point] = net"<<endl<<endl<<endl;
					branch=check_branch(param, nlayers, lparam.name());
				}
				else
				{
					if (br_concat.empty())
						br_concat=find_concat(param, nlayers, lparam.name());
					b_num=ep_send="branch_"+ std::to_string(axis-1);
					if (previous_endpoint.empty() || previous_endpoint != br_concat)
					{		
						axis= 0;
						values = "values=[branch_"+ std::to_string(axis);
						ep_send=current_branch="Branch_"+ std::to_string(axis);
						b_num="branch_"+std::to_string(axis);
						end_point = "end_point = \'"+br_concat+"\'";	
						target<<space<< end_point<<endl;	

						//changed for multiplexing code
						target<<space<< "net = selectinput(end_point, net)"<<endl<<endl;
						target<<space<< "with tf.variable_scope(end_point):"<<endl;
						space=indent(++ind);
					}
					
					if (oldparam.name()!=lparam.bottom()[0])
					{
						//changed for multiplexing code
						if (previous_endpoint.size()!= 0 && previous_endpoint == br_concat)
						{
							select="selectdepth(end_point,";
							close=")";
						}
						ep_send="net";
						if (axis>0)
							values = values + ", branch_" + std::to_string(axis);
						current_branch = "Branch_"+std::to_string(axis);
						b_num="branch_"+std::to_string(axis);
						target<<space<< "with tf.variable_scope(\'"+current_branch+"\'):"<<endl;
						axis++;
					}
					mixed(lparam,ind+1,ep_send,b_num,select, close);
					previous_endpoint = br_concat;
					oldparam = lparam;
				}	
		  	}
			else if (branch==2)
				dropout(lparam,ind);
		}

	}
delete input;
//close(fd);*/
   
   
return 0;
}
