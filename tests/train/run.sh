cd ../..
python setup.py build develop
cd tests/train
CUDA_LAUNCH_BLOCKING=1 python doc_test.py /vol/cc_proposals/images /vol/annotations /vol/cc_proposals/csv
