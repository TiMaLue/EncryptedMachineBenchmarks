(ma6) 
# amin @ amin-xu6 in ~/repos/ma-praxis/Target.TFEncrypted on git:master x [19:50:58] 
$ find /home/amin/miniconda3/envs/ma6/ -name "work_sharder.h" 
/home/amin/miniconda3/envs/ma6/lib/python3.6/site-packages/tensorflow_core/include/tensorflow/core/util/work_sharder.h
(ma6) 
# amin @ amin-xu6 in ~/repos/ma-praxis/Target.TFEncrypted on git:master x [19:51:53] 
$ export CPATH="/home/amin/miniconda3/envs/ma6/lib/python3.6/site-packages/tensorflow_core/include" 
(ma6) 
# amin @ amin-xu6 in ~/repos/ma-praxis/Target.TFEncrypted on git:master x [19:52:06] 
$ cd ../../ma-praxis-related/tf-encrypted 
(ma6) 
# amin @ amin-xu6 in ~/repos/ma-praxis-related/tf-encrypted on git:master x [19:52:29] 
$ pip install
(ma6) 
# amin @ amin-xu6 in ~/repos/ma-praxis-related/tf-encrypted on git:master x [19:52:34] C:130
$ make build
mkdir -p tf_encrypted/operations/secure_random
g++ -std=c++11 -shared operations/secure_random/secure_random.cc -o tf_encrypted/operations/secure_random/secure_random_module_tf_1.15.5.so \
        -fPIC -I/home/amin/miniconda3/envs/ma6/lib/python3.6/site-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=0 -L/home/amin/miniconda3/envs/ma6/lib/python3.6/site-packages/tensorflow_core -l:libtensorflow_framework.so.1 -O2 -I/home/amin/repos/ma-praxis-related/tf-encrypted/build/include -L/home/amin/repos/ma-praxis-related/tf-encrypted/build/lib -lsodium
mkdir -p tf_encrypted/operations/aux
g++ -std=c++11 -shared operations/aux/aux_kernels.cc -o tf_encrypted/operations/aux/aux_module_tf_1.15.5.so \
        -fPIC  -I/home/amin/miniconda3/envs/ma6/lib/python3.6/site-packages/tensorflow_core/include -D_GLIBCXX_USE_CXX11_ABI=0 -L/home/amin/miniconda3/envs/ma6/lib/python3.6/site-packages/tensorflow_core -l:libtensorflow_framework.so.1 -O2
(ma6) 
# amin @ amin-xu6 in ~/repos/ma-praxis-related/tf-encrypted on git:master x [19:52:50] 
$ 
