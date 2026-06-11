# 🚗 tools for get few shot data in nuscenes, navsim and bench2drive

## News

* **` Feb. 01th, 2026`:** Code are released!

### 📁 1. Download NAVSIM Dataset

To obtain the [navsim dataset](https://github.com/autonomousvision/navsim/tree/main):

```bash
bash download/download_maps.sh
bash download/download_navtrain.sh
bash download/download_test.sh
```

---

## ⚙️ 2. Installation

```
please refer to sparsedrive, bench2drivezoo and navsim to get data
```

---

## 🚀 3. run py for getting few shot data

```
python get_ben_fs.py # get ben_fs dataset
python get_navsim_fs.py # get navsim_fs dataset
python get_nus_fs.py # get nus_fs dataset
python get_motion_for_navsim.py # get navsim motion data
```
