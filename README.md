# yolov8-image-analysis
A vanilla implementation of the integration between Streamlit's GUI and a pretrained YOLO model without fine-tuning. It allows the user to upload an image of their liking, and it detects the objects present in the image and outputs a written lits of these objects.
<p align="center">
  <img src="https://github.com/MariamH78/yolov8-image-analysis/assets/99722575/5e03ee55-febe-4025-86a8-8e6dd792e771"/>
</p>

## To run locally:
This assumes that the packages `streamlit`, `numpy`, `opencv` and `ultralytics` are installed.

```sh
streamlit run  script.py --server.enableXsrfProtection false 
```
