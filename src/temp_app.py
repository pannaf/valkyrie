from fastapi import FastAPI, Request, Depends, BackgroundTasks
from src.utils.logtils import user_id_var, LoggingContextManager
from src.small_funcs import g

app = FastAPI()


# Dependency to set the user_id context
def set_user_id(request: Request):
    user_id = request.headers.get("X-User-ID", "anonymous")
    return LoggingContextManager(user_id)


@app.middleware("http")
async def add_logging_context(request: Request, call_next):
    user_id = request.headers.get("X-User-ID", "anonymous")
    with LoggingContextManager(user_id):
        response = await call_next(request)
        return response


@app.get("/run")
def run_g(background_tasks: BackgroundTasks, context_manager: LoggingContextManager = Depends(set_user_id)):
    def background_task():
        with context_manager as logger_context:
            try:
                with logger_context.catch(reraise=False):
                    g()
            finally:
                logger_context.info(f"Completed execution for user {context_manager.user_id}")
                print(f"{user_id_var.get()=}")

    background_tasks.add_task(background_task)
    return {"message": "Function g() is running in the background"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
