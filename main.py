from fastapi import FastAPI
import spam_classification
import classifier
import uvicorn

app = FastAPI()
app.include_router(classifier.router)
app.include_router(spam_classification.router)


@app.get("/")
async def root():
    return {"For interactive API visit": '/docs'}


if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')
