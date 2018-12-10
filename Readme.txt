Run the following command to run the code:

1. g++ -I /usr/include -L /usr/lib code_generator_simple.cpp caffe.pb.cc -lprotobuf -pthread
2. ./a.out

This will produce the inception_v1_simple.py in the same folder.
For the multiplexing code, run the following command:

1. g++ -I /usr/include -L /usr/lib code_generator_multiplexing.cpp caffe.pb.cc -lprotobuf -pthread
2. ./a.out

This will produce the inception_v1_multiplexing.py in the same folder.