library(raster)

crop_to_main_dtm_size <- function(raster_file, main_raster_file) {
  tryCatch({
    # Open the main raster file to get its extent
    main_dataset <- raster(main_raster_file)
    
    if (is.null(main_dataset)) {
      cat("Error: Could not open the main raster file\n")
      return(NULL)
    }
    
    # Get the main raster's extent
    main_raster_extent <- extent(main_dataset)
    
    # Read the source raster file
    src_raster <- raster(raster_file)
    
    # Crop the source raster to the extent of the main raster
    cropped <- crop(src_raster, main_raster_extent)
    
    return(cropped)
  }, 
  error = function(e) {
    cat("Error: ", e$message, "\n")
  })
}
