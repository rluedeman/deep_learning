from multiprocessing.sharedctypes import Value
from fastapi import FastAPI
from fastapi import HTTPException

from gan_pipeline.app import config
from gan_pipeline.app.models import GanPipelineMissingException
from gan_pipeline.app.models import GanPipelineModelBase, GanPipelineModel
from gan_pipeline.app.models import CalibrationImageRequest

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
    model = GanPipelineModel(project_name)
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
        print("Addiong calibration images to {}".format(project_name))
        model = GanPipelineModel(project_name)
        model.add_calibration_images(calibration_request)
        # config.QUEUE.enqueue(model.add_calibration_images, calibration_request)
        return calibration_request
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

