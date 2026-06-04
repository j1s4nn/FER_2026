All output figures and metrics are auto-saved here after running the pipeline.

output/figures/
  - fig_train_curves_cnn.png       Training & validation loss/accuracy for CNN
  - fig_train_curves_mlp.png       Training & validation loss/accuracy for MLP
  - fig_train_curves_fusion.png    Training & validation loss/accuracy for FusionNet
  - fig_confusion_cnn.png          Confusion matrix (CNN)
  - fig_confusion_mlp.png          Confusion matrix (MLP)
  - fig_confusion_fusion.png       Confusion matrix (FusionNet)
  - fig_pz_predictions_*.png       Predicted vs. GT pole-zero map panels
  - fig_f1_per_class_*.png         Per-class F1 bar charts

output/dataset_viz/
  - fig01_sample_pz_maps.png       Sample pole-zero map images per class
  - fig02_mean_freq_response.png   Mean frequency response per class
  - fig03_class_distribution.png   Class balance bar chart
  - fig04_freq_response_heatmap.png Heatmap of frequency responses
  - fig05_pixel_distribution.png   Pixel intensity histogram
  - fig06_pz_scatter.png           Pole-zero scatter on Z-plane

output/test_metrics/
  - test_metrics_summary.csv       Accuracy, Precision, Recall, F1, PSNR, SSIM
  - test_metrics_summary.json      Same data in JSON
  - fig_model_comparison.png       Bar chart comparing all models
  - classification_report_*.txt    Full sklearn classification reports
