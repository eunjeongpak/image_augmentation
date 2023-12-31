# image_augmentation

useful codes related to augmentation

### Last Updated : 2023.08.20

---

## aug.py

### 1. Data Structure

```buildoutcfg
[--example]
├─D
│ └─img_folder
│   └─all_ex
│        01na00ej000001kr.jpg
│        01na00ej000001kr.xml
```

### 2. CLI command 

1) WHEN WE USE AUG.PY

```buildoutcfg
[--h]
cd utils
python aug.py --dir {dir} --new_dir {new_dir} -- method {method}
[--example]
cd utils
python aug.py --dir 'D:/img_folder/all_ex/' --new_dir 'D:/img_folder/img_aug/' --method 'rddc'
```

- dir : original file path
- new_dir : the location where you want to save the results
- method : augmentation method
  - 'rddc': rotate & deep dark color
  - 'rdc': rotate & dark color
  - 'rldc': rotate & little dark color
  - 'rbc': rotate & bright color
  - 'rlc': rotate & light color
  - 'rn': rotate & noise
  - 'rgn': rotate & gaussian noise
  - 'fc': flip & color
  - 'fs': flip & sharpen
  - 'ts': translation & shearing
  - 'crr': crop & resize & rotate

2) WHEN WE USE AUGMENT.PY

```buildoutcfg
[--h]
cd utils
python aug.py --dir {dir} --new_dir {new_dir} 
[--example]
cd utils
python augment.py --dir 'D:/img_folder/all_ex/' --new_dir 'D:/img_folder/img_aug/'
```

- dir : original file path
- new_dir : the location where you want to save the results
