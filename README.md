# Canopy Height Estimation using sinc model

* Step 1: Data search 
  * ALOS-2
    * [Pasco PLATFORM for ALOS-2](https://satpf.jp/spf/?sb=search&sensor=ALOS-2_PALSAR-2)
    * [Australian public LiDAR data (height GT)](http://data.auscover.org.au/xwiki/bin/view/Product+pages/Airborne+Lidar)
* Step 2: Prepare a coherence image for HV polarization (SNAP)
* Step 3: Preprocessing (`preprocessing.ipynb`)
  * Preprocess the GT data (reprojection, shape adjustment, etc)
  * Preprocess the coherence data (reprojection, shape adjustment, etc)
* Step 4: Main analysis (`height_estimation_sinc`)
  * Train and validate the model for height estimation (sinc model)
  * Export the result as a geotiff map

# Workflow height estimation
![image](https://github.com/jutak0228/canopy_height_sinc/assets/159540763/b823890c-8969-48ab-9813-dea6f3864397)
