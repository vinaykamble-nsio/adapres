# Related work notes (verified 2026-11-22)

Each entry: key, what the paper does, how it relates to our paper, URL checked.
Bibliographic fields were checked against the Crossref DOI record (api.crossref.org/works/<doi>) and the listed page.

## Datasets & models

- **yu2020bdd100k**: Introduces BDD100K, a 100K-video driving dataset with labels for ten tasks, including detection and multi-object tracking.
  Relation: our evaluation data (the BDD100K MOT val clips, split into calib/test sets).
  Checked: https://doi.org/10.1109/CVPR42600.2020.00271 (Crossref record)
- **zhu2022visdrone**: Describes the VisDrone benchmark and challenge for drone-view detection and tracking in images and video.
  Relation: a secondary/candidate dataset; cite only if VisDrone results appear in the paper.
  Checked: https://doi.org/10.1109/TPAMI.2021.3119563 (Crossref record)
- **lin2014coco**: Introduces the MS COCO detection and segmentation dataset and its AP evaluation protocol.
  Relation: YOLO11s is COCO-pretrained, and our mAP follows the COCO-style metric.
  Checked: https://doi.org/10.1007/978-3-319-10602-1_48 (Crossref record)
- **jocher2024yolo11**: The Ultralytics YOLO11 software release. There is no paper; the BibTeX is the one Ultralytics recommends.
  Relation: the detector we quantize (YOLO11s).
  Checked: https://docs.ultralytics.com/models/yolo11/ and https://github.com/ultralytics/ultralytics/blob/main/CITATION.cff. Note: the repo CITATION.cff now has a preferred citation for a 2026 YOLO26 arXiv paper (2606.03748); the YOLO11 docs page still gives the @software entry used here.

## Quantization

- **jacob2018quantization**: Proposes an integer-only inference scheme with affine quantization and quantization-aware training.
  Relation: the foundation for the INT8 arithmetic that TensorRT INT8 engines use; static, not per-input.
  Checked: https://doi.org/10.1109/CVPR.2018.00286
- **nagel2021whitepaper**: A survey/tutorial on post-training quantization and quantization-aware training, with practical pipelines.
  Relation: background on PTQ calibration and why INT8 loses accuracy; our INT8 engine is PTQ-calibrated.
  Checked: https://arxiv.org/abs/2106.08295
- **wang2019haq**: Uses reinforcement learning with hardware feedback to choose per-layer bit-widths (mixed precision).
  Relation: different granularity. HAQ fixes precision per layer offline, while we switch the whole engine per frame at run time.
  Checked: https://doi.org/10.1109/CVPR.2019.00881
- **liu2022dqnet**: Adds a lightweight bit-controller, trained with the network, that picks per-layer bit-widths for each input image.
  Relation: the closest image-level analogue of input-adaptive precision. It reports gains from training-time co-design, whereas we use fixed post-training INT8/FP16 engines and find that content cues do not beat random INT8 placement.
  Checked: https://doi.org/10.1109/CVPR52688.2022.01211; https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_Instance-Aware_Dynamic_Neural_Network_Quantization_CVPR_2022_paper.pdf
- **jin2020adabits**: Trains one network whose weight and activation bit-widths can be switched at deployment, using joint training and switchable clipping levels.
  Relation: enables switching precision at run time but does not decide when to switch. Our work studies the decision policy on fixed engines.
  Checked: https://doi.org/10.1109/CVPR42600.2020.00222; https://arxiv.org/abs/1912.09666
- **sun2021videoiq**: VideoIQ uses a lightweight policy network, trained with the recognizer, to choose a numerical precision for each video frame for efficient video recognition.
  Relation: the same per-frame precision-selection problem, but for action recognition with a jointly trained policy. We test the idea for detection with off-the-shelf TensorRT engines and an INT8 budget, and get a negative result.
  Checked: https://doi.org/10.1109/ICCV48922.2021.00728; https://arxiv.org/abs/2108.10394
- **nvidia_tensorrt_docs**: Official TensorRT documentation, covering INT8/FP16 engine building and quantization workflows.
  Relation: the toolchain we use to build our engines.
  Checked: https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html; .../inference-library/work-quantized-types.html (the page has no version or year).

## Adaptive video analytics / adaptive detection

- **chin2019adascale**: AdaScale picks the input image scale for each video frame with a learned regressor, improving both speed and mAP for video object detection.
  Relation: the same per-frame adaptive idea, but the knob is resolution rather than numeric precision. It is a positive result, which contrasts with our finding for precision.
  Checked: https://proceedings.mlsys.org/paper_files/paper/2019/hash/b3ac5aacb06c91fda1af776a677100ab-Abstract.html (and its BibTeX file)
- **jiang2018chameleon**: Chameleon periodically re-profiles video analytics configurations (resolution, frame rate, model) and shares profiles across cameras over time and space.
  Relation: adapts configuration at coarse (segment-level) granularity and stresses profiling cost. Our controller-overhead finding echoes its point that the cost of adaptation itself matters.
  Checked: https://doi.org/10.1145/3230543.3230574
- **xu2020approxdet**: ApproxDet chooses among approximate detection/tracking configurations on mobile GPUs, using frame content and resource contention.
  Relation: a content-aware per-frame/per-segment controller for detection, close to our content-cue controller. Its knobs are not numeric precision.
  Checked: https://doi.org/10.1145/3384419.3431159
- **xu2022litereconfig**: LiteReconfig uses a cost-benefit analyzer to decide which content features are worth computing and which detection branch to run, accounting for switching overhead.
  Relation: most relevant to our negative finding. It explicitly models the cost of extracting content features and of switching, which in our setting outweighs the INT8 saving.
  Checked: https://doi.org/10.1145/3492321.3519577; https://starsthu2016.github.io/projects/litereconfig/litereconfig_eurosys22.html
- **jiang2021remix**: Remix splits high-resolution frames into non-uniform blocks and assigns detectors of different sizes to them under a latency budget, guided by historical frames.
  Relation: a budgeted, content-guided allocation of compute within a frame (spatial granularity), while ours is temporal (which frames get INT8).
  Checked: https://doi.org/10.1145/3447993.3483274; https://www.microsoft.com/en-us/research/?p=763831
- **lin2023dynamicdet**: DynamicDet adds an adaptive router and loss-based exit criterion that send easy images through a cheaper path of a cascaded YOLO detector.
  Relation: input-adaptive compute for a YOLO detector (early exit/routing) rather than precision; its router is trained jointly, unlike our post-hoc controller.
  Checked: https://doi.org/10.1109/CVPR52729.2023.00608; https://arxiv.org/abs/2304.05552

## Energy measurement

- **nvidia_nvml_api**: The official NVML API reference, which includes the device power and energy query functions.
  Relation: the API behind our GPU power/energy readings on the T4.
  Checked: https://docs.nvidia.com/deploy/nvml-api/index.html (the page has no version or year).
- **yang2024gpupower**: Characterizes the NVIDIA built-in power sensor across 70+ GPUs, shows that nvidia-smi/NVML readings can sample only part of runtime on some GPUs, and gives guidance for accurate energy measurement.
  Relation: methodology caveat for our NVML-based per-frame energy numbers, and support for reporting energy over long windows or repeated runs.
  Checked: https://doi.org/10.1109/SC41406.2024.00028; https://arxiv.org/abs/2312.02741
- **henderson2020energy**: Proposes a framework and tool for systematically tracking and reporting the energy and carbon of ML experiments.
  Relation: supports our practice of reporting energy with hardware and config metadata.
  Checked: https://jmlr.org/papers/v21/20-312.html

## Statistics

- **efron1994bootstrap**: The standard textbook on bootstrap resampling and bootstrap confidence intervals.
  Relation: the basis for our bootstrap CIs.
  Checked: https://doi.org/10.1201/9780429246593 (Crossref: Chapman and Hall/CRC, 1994); https://www.routledge.com/An-Introduction-to-the-Bootstrap/Efron-Tibshirani/p/book/9780412042317 (copyright 1994, 456 pp.). The 1993 date in the draft is the commonly cited print year (seen only in library-catalog search snippets), so it is not confirmed on a primary page.
- **cameron2008bootstrap**: Shows that cluster-bootstrap methods (especially the wild cluster bootstrap-t) give more accurate inference than standard cluster-robust errors when there are few clusters.
  Relation: justifies resampling whole clips (clusters) rather than frames for our CIs, since frames within a clip are correlated.
  Checked: https://doi.org/10.1162/rest.90.3.414
