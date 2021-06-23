//var DRC = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filterMetadata('country_co', 'equals', 'CG');

//import your study region shapefile into the GEE assets folder to be used in the download
var studyRegion = studyRegion

//access the Landsat 8 imagery from the GEE catalogue
var landsat8_sr = ee.ImageCollection("LANDSAT/LC08/C01/T1_SR");

//spatially filter the images to your study region
var spatialFiltered = landsat8_sr.filterBounds(studyRegion);
print('spatialFiltered', spatialFiltered);

// temporally filter the imagery for each year in the summer/dry season
var temporalFiltered_2013 = spatialFiltered.filterDate('2013-06-01', '2013-09-30');
print('temporalFiltered', temporalFiltered_2013);

var temporalFiltered_2014 = spatialFiltered.filterDate('2014-06-01', '2014-09-30');
print('temporalFiltered', temporalFiltered_2014);

var temporalFiltered_2015 = spatialFiltered.filterDate('2015-06-01', '2015-09-30');
print('temporalFiltered', temporalFiltered_2015);

var temporalFiltered_2016 = spatialFiltered.filterDate('2016-06-01', '2016-09-30');
print('temporalFiltered', temporalFiltered_2016);

var temporalFiltered_2017 = spatialFiltered.filterDate('2017-06-01', '2017-09-30');
print('temporalFiltered', temporalFiltered_2017);

var temporalFiltered_2018 = spatialFiltered.filterDate('2018-06-01', '2018-09-30');
print('temporalFiltered', temporalFiltered_2018);

var temporalFiltered_2019 = spatialFiltered.filterDate('2019-06-01', '2019-09-30');
print('temporalFiltered', temporalFiltered_2019);

var temporalFiltered_2020 = spatialFiltered.filterDate('2020-06-01', '2020-09-30');
print('temporalFiltered', temporalFiltered_2020);

// Cloud Masking
/**
 * Define the function to mask clouds based on the pixel_qa band of Landsat 8 SR data.
 * @param {ee.Image} image input Landsat 8 SR image
 * @return {ee.Image} cloudmasked Landsat 8 image
 */
 
function maskL8sr(image) {
  // Bits 3 and 5 are cloud shadow and cloud, respectively.
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);
  // Get the pixel QA band.
  var qa = image.select('pixel_qa');
  // Both flags should be set to zero, indicating clear conditions.
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
                 .and(qa.bitwiseAnd(cloudsBitMask).eq(0));
  return image.updateMask(mask);
}

//Apply the cloud mask to stufy region

var dataset_2013 = temporalFiltered_2013.map(maskL8sr);
var dataset_2014 = temporalFiltered_2014.map(maskL8sr);
var dataset_2015 = temporalFiltered_2015.map(maskL8sr);
var dataset_2016 = temporalFiltered_2016.map(maskL8sr);
var dataset_2017 = temporalFiltered_2017.map(maskL8sr);
var dataset_2018 = temporalFiltered_2018.map(maskL8sr);
var dataset_2019 = temporalFiltered_2019.map(maskL8sr);
var dataset_2020 = temporalFiltered_2020.map(maskL8sr);


//Take median composites of visible and infrared bands
var image_2013 = dataset_2013.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_2014 = dataset_2014.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_2015 = dataset_2015.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_2016 = dataset_2016.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_2017 = dataset_2017.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_2018 = dataset_2018.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_2019 = dataset_2019.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);
var image_2020 = dataset_2020.median().select(['B1', 'B3', 'B2','B4','B5','B6','B7']);

//Clip Composities to study region
var image_2013 = image_2013.clip(studyRegion)
var image_2014 = image_2014.clip(studyRegion)
var image_2015 = image_2015.clip(studyRegion)
var image_2016 = image_2016.clip(studyRegion)
var image_2017 = image_2017.clip(studyRegion)
var image_2018 = image_2018.clip(studyRegion)
var image_2019 = image_2019.clip(studyRegion)
var image_2020 = image_2020.clip(studyRegion)


//Display the visible bands of a composite image
Map.addLayer(image_2013, visParams);
var visParams = {
  bands: ['B4', 'B3','B2'],
  min: 0,
  max: 3000,
  gamma: 1.4,
};

//DOWNLOAD THE IMAGERY TO GOOGLE DRIVE

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_2013,
  description: 'DRC_L8_2013_Training_Image',
  scale: 30,
  region: studyRegion,
  fileFormat: "GeoTIFF",
  maxPixels: 1e13,
  formatOptions: {
    cloudOptimized: true
  }
});
/*
//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_2014,
  description: 'PIREDD_Plataue_L8_2014',
  scale: 30,
  region: studyRegion,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_2015,
  description: 'PIREDD_Plataue_L8_2015',
  scale: 30,
  region: studyRegion,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_2016,
  description: 'PIREDD_Plataue_L8_2016',
  scale: 30,
  region: studyRegion,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_2017,
  description: 'PIREDD_Plataue_L8_2017',
  scale: 30,
  region: studyRegion,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_2018,
  description: 'PIREDD_Plataue_L8_2018',
  scale: 30,
  region: studyRegion,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_2019,
  description: 'PIREDD_Plataue_L8_2019',
  scale: 30,
  region: studyRegion,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});

//Export the image, specifying scale and region.
Export.image.toDrive({
  image: image_2020,
  description: 'PIREDD_Plataue_L8_2020',
  scale: 30,
  region: studyRegion,
  fileFormat: "GeoTIFF",
  formatOptions: {
    cloudOptimized: true
  }
});
*/