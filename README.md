1) the folder structure for a dataset :-

dataset 
   --test
      --guns
      --non guns
   --train
      --guns
      --non guns

2) For GPU training :- install CUDA , Nvidia drivers and cuDNN (optional)

3) To train the model :-

4) 1) python3 -m venv .venv
  
   2) pip install librosa joblib numpy matplotlib pydub scipy scikit-learn tensorflow[and-cuda]
  
   3) ensure that dataset folder is present
  
   4) python3 main.py
  
   5) h5, graph and  would be stored in root dir
  
5) For testing model :-

6) 1) edit audio file path that you want to test
  
   2) python3 test.py
