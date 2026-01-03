# Data Directory

Download the flower dataset by running:

```bash
python src/download_data.py
```

Or manually:
```bash
curl -O http://download.tensorflow.org/example_images/flower_photos.tgz
tar -xzf flower_photos.tgz
```

After extraction, the structure will be:
```
data/
├── flower_photos/
│   ├── daisy/
│   ├── dandelion/
│   ├── roses/
│   ├── sunflowers/
│   └── tulips/
└── README.md
```
