
echo Generating RR ...
python src/preprocessing/gendata.py
python src/preprocessing/project_signals_on_sphere.py --input_type RR --lmax 10

echo Generating NRR ...
python src/preprocessing/gendata.py --no_rotate_train
python src/preprocessing/project_signals_on_sphere.py --input_type NRR --lmax 10

echo Generating NRNR ...
python src/preprocessing/gendata.py --no_rotate_train --no_rotate_test
python src/preprocessing/project_signals_on_sphere.py --input_type NRNR --lmax 10
