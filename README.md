# image_augmentation

useful codes related to augmentation

### Last Updated : 2023.08.15

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

```buildoutcfg
[--h]
cd utils
python aug.py --dir {dir} --new_dir {new_dir} -- method {method}
[--example]
cd utils
python aug.py --dir 'D:/img_folder/all_ex/' --new_dir 'D:/img_folder/img_aug/' --method 'fc'
```

- dir : original file path
- new_dir : the location where you want to save the results
- method : augmentation method 
  - 'rn': rotate & noise
  - 'fc': flip & color
  - 'dc': dark color
  - 'bc': bright color
  - 'ts': translation & shearing
  - 'cr': crop & resize & rotate



