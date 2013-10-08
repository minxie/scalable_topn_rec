scalable_topn_rec
=================
* git clone scalable_topn_rec
* git clone https://github.com/GraphChi/graphchi-cpp.git # Main folder
* cd graphchi_cpp
* ./install.sh # Eigen
* git clone https://github.com/minxie/graphchi_mf_topn_index.git myapps

* Running cython
```
source VirtualEnv/bin/activate
cd src
python setup.py build_ext --inplace
cd ..
./sgd.sh
deactivate
```
