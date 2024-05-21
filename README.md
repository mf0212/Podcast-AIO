## 1 Installation

### Linux/Unix

1. Install [Anaconda](https://www.anaconda.com/), Python and `git`.

2. Creating the env and install the requirements.
  ```bash
  git clone https://github.com/mf0212/Podcast-AIO.git

  conda create -n podcast python=3.8

  conda activate podcast

  pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

  conda install ffmpeg

  pip install -r requirements.txt

  ```  


## 2 Download Model

```bash
bash download_models.sh
```

## 3 Quick Start

#### Prepare data



Describe podcast data : 
```
data/
│  ├──podcast/
│    ├──slide-1-guest.wav
│    ├──slide-1-host.wav
│    ├──slide-2-guest.wav
│    ├──slide-2-host.wav
│    ├──......
│    ├──podcast_script.json
│ 
│  ├──slides/
│    ├──1.jpg
│    ├──2.jpg
│    ├──3.jpg
│    ├──......
│ 
│  ├──podcast.png
│  ├──podcast.txt
```

Run `process_audio_podcast.py` to process podcast

```bash
python audio_processing/process_audio_podcast.py \
    --audio_directory data/podcast \
    --output_dir data/podcast/processed_audio \
    --podcast_file_path data/podcast/podcast_script.json
```


Expectation processed podcast data:

```
data/
├──podcast/
│  ├──podcast_script.json
│  │
│  ├──processed_audio/
│      ├──guest.wav
│      ├──host.wav
│      │
│      ├──subtitle/
│      │    ├──1.json
│      │    ├──2.json
│      │    ├──......
├──slides/
│    ├──1.jpg
│    ├──2.jpg
│    ├──3.jpg
│    ├──......
│ 
│ 
├──podcast.png
├──podcast.txt
```



#### Guest / Host Talking

```bash
python inference.py  \
                --driven_audio_folder data/podcast/processed_audio \
                --source_image data/podcast.png\
                --label_file data/podcast.txt \
                --result_dir results/podcast/ \
                --still \
                --preprocess full \
```

**OR RUN**


```bash
bash podcast.sh
```


Check your `results` dir once podcast video has been created
