import rasterio
from rasterio.features import shapes
import geopandas as gpd


def export_tiff(array, target, reference, nodata=None, band=1, dtype=None):

    if dtype is None:
        dtype = array.dtype

    ref = rasterio.open(reference)
    tar = rasterio.open(
        target,
        mode='w',
        driver='GTiff',
        height=ref.height,
        width=ref.width,
        crs=ref.crs,
        transform=ref.transform,
        count=band,
        dtype=dtype,
        nodata=nodata
    )

    tar.write(array, band)
    tar.close()
    ref.close()


def vectorize_raster(array, reference):
    with rasterio.Env():
        with rasterio.open(reference) as src:
            image = src.read(1).astype('uint16')
            result = ({'properties': {'val': s}, 'geometry': s} for i, (s, v) in enumerate(
                shapes(image, mask=array, transform=src.transform)))

    result = gpd.GeoDataFrame.from_features(list(result))

    return result
