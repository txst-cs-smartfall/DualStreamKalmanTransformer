#!/bin/bash
weights="czu_accwKD.pt"
work_dir="exps/czu_woKD/distillation"
result_file="result.txt"

# weights="berkley_best.pt"
# work_dir="exps/bmhad_woKD/late_fusion_epoch150_alldrop0.4"

#Utd student without KD
#python3 main.py --config ./config/utd/student.yaml --work-dir exps/UTD_woKD/MM --device 0 --base-lr 2.5e-2 

#Utd teacher 
#python3 main.py --config ./config/czu/teacher.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True
#python3 main.py --config ./config/utd/teacher.yaml --work-dir $work_dir --weights $work_dir/$weights --device 4  --base-lr 2.5e-3 --phase 'test' --result-file $work_dir/$result_file 


#berkley_student
# python3 main.py --config ./config/berkley/teacher.yaml --work-dir "$work_dir" --model-saved-name "$weights" --device 3 --base-lr 2.5e-3 --include-val True
# python3 main.py --config ./config/berkley/teacher.yaml --work-dir "$work_dir" --weights "$work_dir/$weights" --device 3  --base-lr 2.5e-3 --phase 'test'

#utd student
#python3 main.py --config ./config/czu/student.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True

#distillation 
#python3 distiller.py --config ./config/utd/distill.yaml --work-dir exps/UTD_wKD/MMNorm --device 0 --base-lr 2.5e-3 --include-val True
#python3 distiller.py --config ./config/utd/distill.yaml --weights exps/UTD_wKD/MMNorm/test.pt --device 0 --base-lr 2.5e-3 --phase 'test'

#czu 
python3 distiller.py --config ./config/czu/distill.yaml --work-dir $work_dir --model-saved-name $weights  --weights $work_dir/$weights --device 3 --base-lr 2.5e-3 --include-val True
