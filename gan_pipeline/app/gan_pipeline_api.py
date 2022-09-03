import io
from multiprocessing.sharedctypes import Value
from urllib import response

from cv2 import threshold

from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse

from gan_pipeline.app import config
from gan_pipeline.app.models import GanPipelineMissingException
from gan_pipeline.app.models import GanPipelineModelBase, GanPipelineModel
from gan_pipeline.app.models import CalibrationImageRequest, FilterCalibrationImagesRequest, TrainingImagesRequest

# uvicorn data_utils.services.fast_api.api:app --host 0.0.0.0 --port 5000 --reload
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello, world!"}


#############################
# GanProjects
#############################
@app.post("/gan_projects/")
async def create_gan_project(project_name: str):
    """
    Create a new GanProject.
    """
    model = GanPipelineModel(project_name, create=True)
    return GanPipelineModelBase.from_orm(model)


@app.get("/gan_projects/")
async def get_gan_projects():
    """
    Get a list of all GanProjects.
    """
    return [GanPipelineModelBase.from_orm(GanPipelineModel(name)) for name in config.get_gan_pipeline_datasets()]


#############################
# GanProject
#############################
@app.get("/gan_projects/{project_name}/")
async def get_gan_project(project_name: str):
    """
    Get a GanProject.  
    """
    try:
        return GanPipelineModelBase.from_orm(GanPipelineModel(project_name))
    except GanPipelineMissingException:
        # Return a 404 error if the project doesn't exist
        raise HTTPException(status_code=404, detail="GanPipelineModel {} does not exist.".format(project_name))

@app.post("/gan_projects/{project_name}/calibration_images/")
async def post_calibration_images(project_name: str, calibration_request: CalibrationImageRequest):
    """
    Add calibration images to a GanProject.
    """
    try:
        print("Adding calibration images to {}".format(project_name))
        model = GanPipelineModel(project_name)
        model.add_calibration_images(calibration_request)
        # config.QUEUE.enqueue(model.add_calibration_images, calibration_request)
        return calibration_request
    except GanPipelineMissingException as e:
        # Return a 404 error if the project doesn't exist
        raise HTTPException(status_code=404, detail="GanPipelineModel {} does not exist.".format(project_name)) 


@app.post("/gan_projects/{project_name}/training_images/")
async def post_training_images(project_name: str, training_request: TrainingImagesRequest):
    """
    Add training images to a GanProject.
    """
    try:
        print("Adding training images to {}".format(project_name))
        model = GanPipelineModel(project_name)
        model.add_training_images(training_request)
        # model.add_training_images_parallel(training_request)
        # config.QUEUE.enqueue(model.add_training_images, training_request)
        return training_request
    except GanPipelineMissingException as e:
        # Return a 404 error if the project doesn't exist
        raise HTTPException(status_code=404, detail="GanPipelineModel {} does not exist.".format(project_name)) 


@app.post("/gan_projects/{project_name}/filter_calibration_images/")
async def get_filtered_calibration_images(project_name: str, filter_request: FilterCalibrationImagesRequest):
    try:
        model = GanPipelineModel(project_name)
        return model.filter_calibration_images(filter_request)
    except GanPipelineMissingException as e:
        # Return a 404 error if the project doesn't exist
        raise HTTPException(status_code=404, detail="GanPipelineModel {} does not exist.".format(project_name)) 


@app.get("/gan_projects/{project_name}/calibration_images/{image_id}",
         responses = {
            200: {
                "content": {"image/png": {}}
            }
        },
        response_class = StreamingResponse,
)
async def get_calibration_image(project_name: str, image_id: str):
    try:
        model = GanPipelineModel(project_name)
        try:
            # Return the image as bytes
            image_id = image_id.replace('.jpg', '').replace('.png', '')
            img = model.get_calibration_image_with_id(image_id)
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="JPEG")
            img_bytes.seek(0)
            return StreamingResponse(content=img_bytes, media_type="image/jpg")
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail="Calibration image {} does not exist.".format(image_id))

    except GanPipelineMissingException as e:
        # Return a 404 error if the project doesn't exist
        raise HTTPException(status_code=404, detail="GanPipelineModel {} does not exist.".format(project_name))


#############################
# HTML Views
#############################
@app.get("/gan_projects/{project_name}/filtered_calibration_images/", response_class=HTMLResponse)
async def get_filtered_calibration_images_html(
    project_name: str, min_threshold: float = 0.85, max_threshold: float = 1.0, num_images: int = 100):
    """
    Return an HTML page with a list of filtered calibration images.
    """
    try:
        print("Filtering calibration images for {}".format(project_name))
        model = GanPipelineModel(project_name)
        request = FilterCalibrationImagesRequest(min_threshold=min_threshold, max_threshold=max_threshold, num_images=num_images)
        filtered_response = model.filter_calibration_images(request)
        # Reverse the url for the calibration image
        url_base = f"/gan_projects/{project_name}/calibration_images/"
        html_images = [f"<img src='{url_base}{path}' />".format() for path in filtered_response.images]
        html_content = """<html><body>{}</body></html>""".format("\n".join(html_images))
        return HTMLResponse(content=html_content)
    except GanPipelineMissingException as e:
        # Return a 404 error if the project doesn't exist
        raise HTTPException(status_code=404, detail="GanPipelineModel {} does not exist.".format(project_name))


#############################
# Test
#############################
from gan_pipeline.app.tasks import count_words_at_url

@app.get("/test/")
async def test_queue():
    job = config.QUEUE.enqueue(count_words_at_url, 'http://nvie.com/')
    return job.get_id()

