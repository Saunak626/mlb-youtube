# MLB-YouTube Dataset

The MLB-YouTube dataset is a new, large-scale dataset consisting of 20 baseball games from the 2017 MLB post-season available on YouTube with over 42 hours of video footage. Our dataset consists of two components: segmented videos for activity recognition and continuous videos for activity classification. Our dataset is quite challenging as it is created from TV broadcast baseball games where multiple different activities share the camera angle. Further, the motion/appearance difference between the various activities is quite small.

Please see our paper for more details on the dataset \[[arXiv](https://arxiv.org/abs/1804.03247)\].

If you use our dataset or find the code useful for your research, please cite our paper:

```
    @inproceedings{mlbyoutube2018,
              title={Fine-grained Activity Recognition in Baseball Videos},
	      booktitle={CVPR Workshop on Computer Vision in Sports},
	      author={AJ Piergiovanni and Michael S. Ryoo},
	      year={2018}
    }
```

Example Frames from various activities:
![Examples](/examples/mlb-youtube-github.png?raw=true "Examples")

# NEW! MLB-YouTube Captions

We densely annotated the videos with captions from the commentary given by the announcers, resulting in approximately 50 hours of matching text and video. These captions only roughly describe what is happening in the video, and often contain unrelated stories or commentary on a previous event, making this a challenging task.
Examples of the text and video:
![Examples](/examples/mlb-youtube-captions-github.png?raw=true "Examples")

For more details see our paper introducing the captions dataset \[[arXiv](https://arxiv.org/abs/1806.08251)\].

```
  @article{mlbcaptions2018
        title={Learning Shared Multimodal Embeddings with Unpaired Data},
	author={AJ Piergiovanni and Michael S. Ryoo},
        journal={arXiv preprint arXiv:1806.08251},
        year={2018}
}
```

# Segmented Dataset

Our segmented video dataset consists of 4,290 video clips. Each clip is annotated with the various baseball activities that occur, such as swing, hit, ball, strike, foul, etc. A video clip can contain multiple activities, so we treat this as a multi-label classification task. A full list of the activities and the number of examples of each is shown in the table below.

| Activity     | \# Examples |
| ------------ | ----------- |
| No Activity  | 2983        |
| Ball         | 1434        |
| Strike       | 1799        |
| Swing        | 2506        |
| Hit          | 1391        |
| Foul         | 718         |
| In Play      | 679         |
| Bunt         | 24          |
| Hit by Pitch | 14          |

We additionally annotated each clip containing a pitch with the pitch type (e.g., fastball, curveball, slider, etc.) and the speed of the pitch. We also collected a set of 2,983 hard negative examples where no action occurs. These examples include views of the crowd, the field, or the players standing before or after a pitch occurred. Examples of the activities and hard negatives are shown below:

### Strike

<img src="/examples/strike1.gif?raw=true" width="425"> <img src="/examples/strike2.gif?raw=true" width="425">

### Ball

<img src="/examples/ball1.gif?raw=true" width="425"> <img src="/examples/ball2.gif?raw=true" width="425">

### Swing

<img src="/examples/swing1.gif?raw=true" width="425"> <img src="/examples/swing2.gif?raw=true" width="425">

### Hit

<img src="/examples/hit1.gif?raw=true" width="425"> <img src="/examples/hit2.gif?raw=true" width="425">

### Foul

<img src="/examples/foul1.gif?raw=true" width="425"> <img src="/examples/foul2.gif?raw=true" width="425">

### Bunt

<img src="/examples/bunt1.gif?raw=true" width="425"> <img src="/examples/bunt2.gif?raw=true" width="425">

### Hit By Pitch

<img src="/examples/hbp1.gif?raw=true" width="425"> <img src="/examples/hbp2.gif?raw=true" width="425">

### Hard-Negative No Activity

<img src="/examples/neg1.gif?raw=true" width="425"> <img src="/examples/neg2.gif?raw=true" width="425">

# Continuous Dataset

Our continuous video dataset consists of 2,128 1-2 minute long clips from the videos. We densely annotate each frame of the clip with the baseball activities that occur. Each continuous clip contains on average of 7.2 activities, resulting in a total of over 15,000 activity instances. We evaluate models using per-frame mean average precision (mAP).

# Create the dataset

1. Download the youtube videos. Run `python download_videos.py` which relies on youtube-dl. Change the `save_dir` in the script to where you want the full videos saved.
2. To extract the segmented video clips, run `python extract_segmented_videos.py` and change `input_directory` to be the directory containing the full videos and `output_directory` to be the location to save the extracted clips.
3. To extract the continuous video clips, run `python extract_continuous_videos.py` and change `input_directory` to be the directory containing the full videos and `output_directory` to be the location to save the extracted clips.

# 中文使用指南

本部分提供详细的中文指南，帮助您下载和处理数据集。

## 1. 下载原始视频 (download_videos.py)

这个脚本的最终目标是下载本项目所需的 **20 个原始、完整的 YouTube 比赛视频**。请按照以下步骤操作：

### 1.1 环境准备

1.  **激活虚拟环境**: 确保您在终端中已经激活了 Python 虚拟环境。
2.  **安装 `yt-dlp`**: `yt-dlp` 是 `youtube-dl` 的一个分支，功能更强大且更新更频繁。如果尚未安装，请运行：
    ```bash
    pip install yt-dlp
    ```
    _注意：原 `README` 中提到的 `youtube-dl` 已不再积极维护，建议使用 `yt-dlp`。_

### 1.2 获取并放置 YouTube Cookie (关键步骤)

为了绕过 YouTube 的机器人验证，您需要提供一个有效的 Cookie 文件。

1.  **安装浏览器插件**: 在您的 Chrome 或 Firefox 浏览器中，安装一个可以导出 `cookies.txt` 格式的插件，例如 **`Get cookies.txt LOCALLY`**。
2.  **导出 Cookie 文件**:
    - 在浏览器中打开 `https://www.youtube.com` 并确保您已登录。
    - 点击浏览器右上角的 Cookie 插件图标。
    - 点击 **"Export"** 或 **"Export As"** 按钮。
    - 将文件保存到项目 (`mlb-youtube`) 的根目录下，并**必须命名为 `youtube-cookies.txt`**。

> **重要提示**: 这个 Cookie 文件会过期。如果下载失败并提示 `Sign in to confirm you're not a bot`，请**重复此步骤**，用新的 Cookie 文件覆盖旧的即可。

### 1.3 运行下载脚本

1.  打开终端，`cd` 到 `mlb-youtube` 项目的根目录。
2.  运行以下指令开始下载：
    ```bash
    python download_videos.py
    ```

#### 脚本功能

- **智能扫描**: 自动跳过 `/videos` 目录中已完整下载的视频。
- **断点续传**: 自动从 `.part` 文件处继续下载未完成的视频。
- **进度显示**: 显示总进度 `[X/20]`。
- **超时处理**: 为每个视频设置了下载超时。

---

## 2. 理解数据标注 (JSON 文件)

项目中的 `data/*.json` 文件是数据集的灵魂，它们定义了每个视频剪辑的时间戳和行为标签。

### 2.1 `mlb-youtube-segmented.json`

- **用途**: 用于**分割短视频片段**，服务于**动作识别** (Activity Recognition) 任务。
- **内容**: 包含数千个短视频片段的定义。每个片段都包含：
  - `url`: 原始 YouTube 视频地址。
  - `start`, `end`: 片段在原始视频中的**秒数**起止时间。
  - `labels`: 核心标签！一个列表，包含此片段中发生的所有事件，如 `['ball', 'strike', 'swing']`。这是一个**多标签**分类问题。
  - `subset`: 数据集划分 (`training`, `testing`, `validation`)。
  - `type`: 更精细的投球类型标签，如 `slider` (滑球)。
  - `speed`: 投球速度，用于回归任务。

### 2.2 `mlb-youtube-continuous.json`

- **用途**: 用于**分割 1-2 分钟的连续长视频**，服务于**动作检测** (Activity Detection) 任务。
- **内容**: 定义了 2000 多个 1-2 分钟的连续片段。与 `segmented` 不同，它的标签是**逐帧**标注的，指明了在某个长片段内部，从第 X 帧到第 Y 帧发生了什么具体动作。

### 2.3 `mlb-youtube-captions.json`

- **用途**: 用于**多模态学习**，即连接视频画面与自然语言。
- **内容**: 将视频片段与解说员的**评论字幕**匹配起来。这里的"标签"是描述性的文本，而不是简单的分类词。

### 2.4 `mlb-youtube-negative.json`

- **用途**: 提供**困难负样本 (Hard Negative Examples)**。
- **目的**: 这些是精心挑选的、与真实动作场景非常相似但**未发生关键动作**的片段（如观众席、球员站立等）。使用这些样本训练模型，可以显著降低误报率，让模型学会区分细微差别，变得更"聪明"。

---

## 3. 分割视频与匹配标签 (extract_segmented_videos.py)

下载完所有原始视频后，运行此脚本来创建用于模型训练的短视频数据集。

### 3.1 工作原理

脚本会读取 `mlb-youtube-segmented.json` 文件，并对每个条目执行以下操作：

1.  **定位视频**: 根据 `url` 找到本地对应的原始视频文件。
2.  **精确定位**: 使用 `start` 和 `end` 秒数作为剪辑的起止点。时间戳之所以是高精度小数，是为了确保**帧级别的精确度**。
3.  **提取标签**: 读取 `labels` 列表作为视频片段的标签。
4.  **执行剪辑**: 调用 `ffmpeg` 工具剪辑视频，并以唯一 ID 命名保存。
5.  **生成标签文件**: 创建一个映射，将新生成视频文件的文件名与其标签关联起来。

### 3.2 如何运行

1.  **确保 `ffmpeg` 已安装**：视频分割依赖此工具。
2.  **修改脚本路径**: 打开 `extract_segmented_videos.py`，根据需要修改 `input_directory` (应指向 `videos/`) 和 `output_directory`。
3.  **运行脚本**:
    `bash
python extract_segmented_videos.py
`
    完成后，您将在 `output_directory` 中得到几千个带有明确标签的短视频，可直接用于模型训练。

# Baseline Experiments

We compared many approaches using I3D [1] and InceptionV3 [2] as feature extractors.

## Segmented Video Activity Recognition Results

Please see our paper for more experimental details and results.

Results from multi-label video classification:

| Method                        | mAP (%)  |
| ----------------------------- | -------- |
| Random                        | 16.3     |
| I3D + max-pool                | 57.2     |
| I3D + pyramid pooling         | 58.7     |
| I3D + LSTM                    | 53.1     |
| I3D + temporal conv           | 58.4     |
| I3D + sub-events [3]          | 61.3     |
| IncetpitonV3 + max-pool       | 54.4     |
| InceptionV3 + pyramid pooling | 55.3     |
| InceptionV3 + LSTM            | 57.7     |
| InceptionV3 + temporal conv   | 56.1     |
| InceptionV3 + sub-events [3]  | **62.6** |

Pitch Speed Regression:

| Method                        | RMSE        |
| ----------------------------- | ----------- |
| I3D                           | 4.3 mph     |
| I3D + LSTM                    | 4.1 mph     |
| I3D + sub-events [3]          | 3.9 mph     |
| IncetpitonV3                  | 5.3 mph     |
| IncetpitonV3 + LSTM           | 4.5 mph     |
| IncetpitonV3 + sub-events [3] | **3.6 mph** |

## Continuous Video Activity Detection

| Method                                    | mAP (%)  |
| ----------------------------------------- | -------- |
| Random                                    | 13.4     |
| IncetpitonV3                              | 31.9     |
| IncetpitonV3 + max-pool                   | 35.2     |
| InceptionV3 + pyramid pooling             | 36.8     |
| InceptionV3 + LSTM                        | 34.1     |
| InceptionV3 + temporal conv               | 33.4     |
| InceptionV3 + sub-events [3]              | 37.3     |
| InceptionV3 + super-events [4]            | 39.6     |
| InceptionV3 + sub+super-events            | 40.9     |
| InceptionV3 + TGM [5]                     | 37.4     |
| InceptionV3 + 3 TGM [5]                   | 38.2     |
| InceptionV3 + super-event [4] + 3 TGM [5] | 42.9     |
| I3D                                       | 34.2     |
| I3D + max-pool                            | 36.8     |
| I3D + pyramid pooling                     | 39.7     |
| I3D + LSTM                                | 39.4     |
| I3D + temporal conv                       | 39.2     |
| I3D + sub-events [3]                      | 38.5     |
| I3D + super-events [4]                    | 39.1     |
| I3D + sub+super-events                    | 40.4     |
| I3D + TGM [5]                             | 38.5     |
| I3D + 3 TGM [5]                           | 40.1     |
| I3D + super-event [4] + 3 TGM [5]         | **47.1** |

# Experiments

We provide our code to train and evalute the models in the experiments directory. We have the various models implemented in [models.py](/experiments/models.py), a script to load the dataset, and a script to train the models as well.

We also include our PyTorch implementation of I3D, see [pytorch-i3d](https://github.com/piergiaj/pytorch-i3d) for more details.

# Requirements

- [youtube-dl](https://rg3.github.io/youtube-dl/) to download the videos
- tested with ffmpeg 2.8.11 to extract clips
- PyTorch (tested with version 0.3.1)

# References

[1] J. Carreira and A. Zisserman. Quo vadis, action recognition? A new model and the kinetics dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017. \[[arxiv](https://arxiv.org/abs/1705.07750)\] \[[code](https://github.com/deepmind/kinetics-i3d)\]

[2] C. Szegedy, V. Vanhoucke, S. Ioffe, J. Shlens, and Z. Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE Conference on Computer Visionand Pattern Recognition (CVPR), 2016

[3] A. Piergiovanni, C. Fan, and M. S. Ryoo. Learning latent sub-events in activity videos using temporal attention filters. In Proceedings of the American Association for Artificial Intelligence (AAAI), 2017 \[[arxiv](https://arxiv.org/abs/1605.08140)\] \[[code](https://github.com/piergiaj/latent-subevents)\]

[4] A. Piergiovanni and M. S. Ryoo. Learning latent super-events to detect multiple activities in videos. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018 \[[arxiv](https://arxiv.org/abs/1712.01938)\] \[[code](https://github.com/piergiaj/super-events-cvpr18)\]

[5] A. Piergiovanni and M. S. Ryoo. Temporal Gaussian Mixture Layer for Videos. arXiv preprint arXiv:1803.06316, 2018 \[[arxiv](https://arxiv.org/abs/1803.06316)\]
