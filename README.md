# CrackSegDiff: Diffusion Probability Model-based Multi-modal Crack Segmentation

## Abstract

Integrating grayscale and depth data in road inspection robots could enhance the accuracy, reliability, and comprehensiveness of road condition assessments, leading to improved maintenance strategies and safer infrastructure. However, these data sources are often compromised by significant background noise from the pavement. Recent advancements in Diffusion Probabilistic Models (DPM) have demonstrated remarkable success in image segmentation tasks, showcasing potent denoising capabilities, as evidenced in studies like SegDiff. Despite these advancements, current DPM-based segmentors do not fully capitalize on the potential of original image data. In this paper, we propose a novel DPM-based approach for crack segmentation, named CrackSegDiff, which uniquely fuses grayscale and range/depth images. This method enhances the reverse diffusion process by intensifying the interaction between local feature extraction via DPM and global feature extraction. Unlike traditional methods that utilize Transformers for global features, our approach employs Vm-unet to efficiently capture long-range information of the original data. The integration of features is further refined through two innovative modules: the Channel Fusion Module (CFM) and the Shallow Feature Compensation Module (SFCM). Our experimental evaluation on the three-class crack image segmentation tasks within the FIND dataset demonstrates that CrackSegDiff outperforms state-of-the-art methods, particularly excelling in the detection of shallow cracks.

Paper: [arxiv](https://arxiv.org/abs/2410.08100)

## A Quick Overview 

## Quantization Results of CrackSegDiff

<div align="center">

  <img width=680 src="https://github.com/sky-visionX/CrackSegDiff/blob/CrackSegdiff/1.png">
    <p><em>The overall architecture of CrackSegDiff.</em></p>
    
  <table>
    <thead>
      <tr>
        <th rowspan="2">模型</th>
        <th colspan="3">Raw intensity</th>
        <th colspan="3">Raw range</th>
        <th colspan="3">Fused raw image</th>
      </tr>
      <tr>
        <th>F1 score</th>
        <th>IoU</th>
        <th>BF score</th>
        <th>F1 score</th>
        <th>IoU</th>
        <th>BF score</th>
        <th>F1 score</th>
        <th>IoU</th>
        <th>BF score</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><a href="https://www.sciencedirect.com/science/article/pii/S0926580522005489">DenseCrack</a></td>
        <td><em>68.2%</em></td>
        <td>56.5%</td>
        <td>-</td>
        <td>78.4%</td>
        <td>65.3%</td>
        <td>-</td>
        <td>81.5%</td>
        <td>69.7%</td>
        <td>-</td>
      </tr>
      <tr>
        <td><a href="https://www.sciencedirect.com/science/article/pii/S0926580522005489">SegNet-FCN</a></td>
        <td><em>75.0%</em></td>
        <td>63.4%</td>
        <td>-</td>
        <td>81.1%</td>
        <td>68.6%</td>
        <td>-</td>
        <td>84.0%</td>
        <td>72.9%</td>
        <td>-</td>
      </tr>
      <tr>
        <td><a href="https://www.sciencedirect.com/science/article/pii/S0926580522005489">CrackFusionNet</a></td>
        <td><em>77.8%</em></td>
        <td>66.5%</td>
        <td>-</td>
        <td>82.6%</td>
        <td>71.3%</td>
        <td>-</td>
        <td>86.8%</td>
        <td>77.3%</td>
        <td>-</td>
      </tr>
      <tr>
        <td><a href="https://www.sciencedirect.com/science/article/pii/S0926580522005489">Unet-fcn</a></td>
        <td>80.57%</td>
        <td>71.25%</td>
        <td>84.44%</td>
        <td>84.86%</td>
        <td>74.69%</td>
        <td>87.44%</td>
        <td>89.84%</td>
        <td>82.53%</td>
        <td>91.56%</td>
      </tr>
      <tr>
        <td><a href="https://github.com/HRNet/HRNet-Semantic-Segmentation">HRNet-OCR</a></td>
        <td>78.55%</td>
        <td>67.73%</td>
        <td>85.13%</td>
        <td>84.89%</td>
        <td>74.18%</td>
        <td>89.47%</td>
        <td>85.07%</td>
        <td>75.55%</td>
        <td>90.05%</td>
      </tr>
      <tr>
        <td><a href="https://github.com/zZhiG/crackmer">Crackmer</a></td>
        <td>76.54%</td>
        <td>64.92%</td>
        <td>81.48%</td>
        <td>81.78%</td>
        <td>69.72%</td>
        <td>84.79%</td>
        <td>87.32%</td>
        <td>78.25%</td>
        <td>89.93%</td>
      </tr>
      <tr>
        <td><a href="https://github.com/HqiTao/CT-crackseg">CT-CrackSeg</a></td>
        <td><em>83.55%</em></td>
        <td>74.39%</td>
        <td><em>88.61%</em></td>
        <td>88.51%</td>
        <td>80.17%</td>
        <td>91.85%</td>
        <td>92.75%</td>
        <td>87.06%</td>
        <td>95.03%</td>
      </tr>
      <tr>
        <td><a href="https://github.com/MedicineToken/MedSegDiff">MedSegDiff</a></td>
        <td>83.05%</td>
        <td><em>74.61%</em></td>
        <td>88.21%</td>
        <td><em>90.87%</em></td>
        <td><em>83.70%</em></td>
        <td><em>92.98%</em></td>
        <td><em>95.03%</em></td>
        <td><em>90.77%</em></td>
        <td><em>96.50%</em></td>
      </tr>
      <tr>
        <td><strong>CrackSegDiff (Ours)</strong></td>
        <td><strong>84.59%</strong></td>
        <td><strong>77.31%</strong></td>
        <td><strong>89.23%</strong></td>
        <td><strong>92.18%</strong></td>
        <td><strong>86.11%</strong></td>
        <td><strong>93.71%</strong></td>
        <td><strong>95.58%</strong></td>
        <td><strong>91.90%</strong></td>
        <td><strong>96.63%</strong></td>
      </tr>
    </tbody>
  </table>
  <p><em>Comparison of CrackSegDiff with state-of-the-art grayscale and depth fused segmentors on the FIND Dataset.</em></p>
  
  <img width=680 src="https://github.com/sky-visionX/CrackSegDiff/blob/CrackSegdiff/2.png">
    <p><em>Qualitative comparison of CrackSegDiff with state-of-the-art segmentation methods. From left to right, the metrics used are F1-Score, IoU, and BF-Score.</em></p>
</div>

## 1.Requirement

``pip install -r requirement.txt``

## 2. Prepare the pre_trained weights

- The weights of the pre-trained VMamba could be downloaded [here](https://github.com/MzeroMiko/VMamba) or [Baidu](https://pan.baidu.com/s/1ci_YvPPEiUT2bIIK5x8Igw?pwd=wnyy). After that, the pre-trained weights should be stored in './pretrained_weights/'.

## Example Cases
### pixel-level ground truth crack data for deep learning-based crack segmentation
1. Download FIND dataset from https://zenodo.org/records/6383044.
  
2. For training, run: ``python CrackSegDiff/segmentation_train.py --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 5e-5 --batch_size 8``

3. For sampling, run: ``python CrackSegDiff/segmentation_sample.py  --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 1``

4. For evaluation, run ``python CrackSegDiff/segmentation_env.py --inp_pth XXX  --out_pth XXX``

In default, the samples will be saved at `` ./results/`` 

## Thanks
Code copied a lot from [MedSegDiff](https://github.com/MedicineToken/MedSegDiff), [guided-diffusion](https://github.com/openai/guided-diffusion), [SegDiff](https://github.com/tomeramit/SegDiff), [VM-UNet](https://github.com/JCruan519/VM-UNet), 
## Cite
Please cite
~~~
@inproceedings{Jiang2024CrackSegDiffDP,
  title={CrackSegDiff: Diffusion Probability Model-based Multi-modal Crack Segmentation},
  author={Xiaoyan Jiang and Licheng Jiang and Anjie Wang and Kaiying Zhu and Yongbin Gao},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:273233791}
}
~~~
