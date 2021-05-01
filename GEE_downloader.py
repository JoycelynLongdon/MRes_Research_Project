#create function to download the image or imagecollection as you desire
def downloader(ee_object,region):
    try:
        #download image
        if isinstance(ee_object, ee.image.Image):
            print('Its Image')
            url = ee_object.getDownloadUrl({
                    'scale': 30,
                    'crs': 'EPSG:3857',
                    'region': region
                })
            return url

        #download imagecollection
        elif isinstance(ee_object, ee.imagecollection.ImageCollection):
            print('Its ImageCollection')
            ee_object_new = ee_object.mosaic()
            url = ee_object_new.getDownloadUrl({
                    'scale': 30,
                    'crs': 'EPSG:3857',
                    'region': region
                })
            return url
    except:
        print("Could not download")
