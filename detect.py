import ee
ee.Initialize(project='academic-oath-507004-i5')

aoi = ee.Geometry.Rectangle([78.0, 30.4, 78.3, 30.7])
collection = ee.ImageCollection('COPERNICUS/S1_GRD') \
    .filterBounds(aoi) \
    .filter(ee.Filter.eq('instrumentMode', 'IW')) \
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
    .select('VV')

c1 = collection.filterDate('2026-08-05', '2026-08-15')
c2 = collection.filterDate('2026-08-18', '2026-08-28')

print('Images in window 1:', c1.size().getInfo())
print('Images in window 2:', c2.size().getInfo())

img1 = c1.mosaic().clip(aoi)
img2 = c2.mosaic().clip(aoi)
db1 = img1.log10().multiply(10).focal_median(30, 'circle', 'meters')
db2 = img2.log10().multiply(10).focal_median(30, 'circle', 'meters')
diff = db2.subtract(db1).rename('change')
change_mask = diff.abs().gt(4)

vectors = change_mask.selfMask().reduceToVectors(
    geometry=aoi, scale=50, geometryType='polygon', maxPixels=1e9, bestEffort=True
)

geojson = vectors.getInfo()
print('Number of detections:', len(geojson['features']))
print(geojson)