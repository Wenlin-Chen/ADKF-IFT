python -u -m pdb -c c main_adkf.py --epochs 2000 --eval_steps 10 --pretrained 0 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset muv --test-dataset muv --seed 0 --gpu_id 0
python -u -m pdb -c c main_adkf.py --epochs 2000 --eval_steps 10 --pretrained 0 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset sider --test-dataset sider --seed 0 --gpu_id 0
python -u -m pdb -c c main_adkf.py --epochs 2000 --eval_steps 10 --pretrained 0 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset tox21 --test-dataset tox21 --seed 0 --gpu_id 0
python -u -m pdb -c c main_adkf.py --epochs 2000 --eval_steps 10 --pretrained 0 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset toxcast --test-dataset toxcast --seed 0 --gpu_id 0

python -u -m pdb -c c main_adkf.py --epochs 2000 --eval_steps 10 --pretrained 1 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset muv --test-dataset muv --seed 0 --gpu_id 0
python -u -m pdb -c c main_adkf.py --epochs 2000 --eval_steps 10 --pretrained 1 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset sider --test-dataset sider --seed 0 --gpu_id 0
python -u -m pdb -c c main_adkf.py --epochs 2000 --eval_steps 10 --pretrained 1 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset tox21 --test-dataset tox21 --seed 0 --gpu_id 0
python -u -m pdb -c c main_adkf.py --epochs 2000 --eval_steps 10 --pretrained 1 --n-shot-train 10 --n-shot-test 10 --n-query 32 --dataset toxcast --test-dataset toxcast --seed 0 --gpu_id 0