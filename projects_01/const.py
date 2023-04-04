from nbox import logger

# when you are running on the NBX platform you don't really need to pass the project_id
# all of it is automatically handled by the platform, however we are providing it here for
# the sake of the example

project_id = "5b363cc7"
logger.info(f"Loading project: {project_id}")
