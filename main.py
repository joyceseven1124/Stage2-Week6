from fastapi import FastAPI, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model import predict_text
import pandas as pd
import os.path

# 定義輸入數據的結構
class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    feedback: str


# 初始化 FastAPI 應用
app = FastAPI()

# 掛載靜態檔案目錄
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.get("/api/model/prediction")
async def predict(title: str = Query(..., description="The title entered by the user")):
    prediction = predict_text(title)
    # 這裡先返回假數據，之後再整合實際的模型
    return {"prediction": prediction, "message": "預測完成"}

@app.post("/api/model/feedback")
async def feedback(input_data: FeedbackInput):
    # 建立一筆回饋資料（字典）
    data = {
        "title": input_data.text,
        "board": input_data.feedback
    }
    # 轉成 DataFrame 再寫入 CSV
    df = pd.DataFrame([data])
    df.to_csv("user-labeled-titles.csv", mode='a', index=False, header=not os.path.exists("user-labeled-titles.csv"), encoding='utf-8-sig')

    return {"message": "Feedback saved!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    # uvicorn.run(app, host="0.0.0.0", port=8000)