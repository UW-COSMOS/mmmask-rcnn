cd ../..
python setup.py build develop
cd tests/train
python  doc_test.py /vol/cc_proposals/images /vol/annotations /vol/cc_proposals/csv
