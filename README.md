## Code for CR-GLoCo: Cross-Resolution Learning via Global–Local Context Consistency for Semi-Supervised 3D Medical Segmentation


# Usage


1. Train the 3d model
```
python train.py --config configs/train_config_3d.yaml
```

4. Test the model 
```
python test.py --model_path /path/to/trained_model
```
# Acknowledgement
Thanks [SSL4MIS](https://github.com/HiLab-git/SSL4MIS) for their wonderfurl work. Part of the code is borrowed from them. Please feel free to cite their work:
```
  @article{media2022urpc,
  title={Semi-Supervised Medical Image Segmentation via Uncertainty Rectified Pyramid Consistency},
  author={Luo, Xiangde and Wang, Guotai and Liao, Wenjun and Chen, Jieneng and Song, Tao and Chen, Yinan and Zhang, Shichuan, Dimitris N. Metaxas, and Zhang, Shaoting},
  journal={Medical Image Analysis},
  volume={80},
  pages={102517},
  year={2022},
  publisher={Elsevier}}
  
  @inproceedings{luo2021ctbct,
  title={Semi-supervised medical image segmentation via cross teaching between cnn and transformer},
  author={Luo, Xiangde and Hu, Minhao and Song, Tao and Wang, Guotai and Zhang, Shaoting},
  booktitle={International Conference on Medical Imaging with Deep Learning},
  pages={820--833},
  year={2022},
  organization={PMLR}}

  @InProceedings{luo2021urpc,
  author={Luo, Xiangde and Liao, Wenjun and Chen, Jieneng and Song, Tao and Chen, Yinan and Zhang, Shichuan and Chen, Nianyong and Wang, Guotai and Zhang, Shaoting},
  title={Efficient Semi-supervised Gross Target Volume of Nasopharyngeal Carcinoma Segmentation via Uncertainty Rectified Pyramid Consistency},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2021},
  year={2021},
  pages={318--329}}
   
  @InProceedings{luo2021dtc,
  title={Semi-supervised Medical Image Segmentation through Dual-task Consistency},
  author={Luo, Xiangde and Chen, Jieneng and Song, Tao and  Wang, Guotai},
  journal={AAAI Conference on Artificial Intelligence},
  year={2021},
  pages={8801-8809}}
  
  @misc{ssl4mis2020,
  title={{SSL4MIS}},
  author={Luo, Xiangde},
  howpublished={\url{https://github.com/HiLab-git/SSL4MIS}},
  year={2020}}
# SSL-Seg
