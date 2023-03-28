
after loading the cliff image
```
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
```
Change focal_length in /common/mocap_dataset.py
line 36
```
focal_length = $ACTUAL_VALUE
```

running command is
```
python demo.py --ckpt data/ckpt/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt --backbone hr48 --input_path $PATH/TO/MP4 --input_type video --multi --infill --smooth --save_results --make_video --frame_rate 30 --gpu $GPU.NO.
```
If start frame has only one person, may fail with --multi option. Depending on the error complaining,
1. comment demo.py line 162
```
ids, bboxes = (list(t) for t in zip(*sorted(zip(ids, bboxes))))
```
2. or remove option --infill
Consider using parallel to run python process in Alvis,
example code is 
```
echo "python3 script1.py" > commands.txt
echo "python3 script2.py" >> commands.txt
echo "python3 script3.py" >> commands.txt
parallel -j 2 --joblog joblog.log < commands.txt

```

TODO:
rebuild clean singularity image
