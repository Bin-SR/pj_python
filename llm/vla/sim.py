from mj_env import my_mujoco_env

import cv2
from openai import OpenAI

import base64
import numpy as np
from io import BytesIO
from PIL import Image

def ndarray_to_base64(img):

    # MuJoCo通常返回RGB
    img = Image.fromarray(img.astype(np.uint8))

    buffer = BytesIO()

    img.save(buffer, format="JPEG")

    img_bytes = buffer.getvalue()

    img_base64 = base64.b64encode(img_bytes).decode("utf-8")

    return img_base64

def ask_llm(img):

    client = OpenAI(
        api_key="sk-97cec811f742451abc2de02f0a5eb7ed",
        # base_url="https://api.deepseek.com"
    )

    img_base64 = ndarray_to_base64(img)

    response = client.chat.completions.create(
            model = "gpt-4o",
            # model="deepseek-chat",
        messages=[
            {
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text":"只告诉我你看到了什么？"
                    },
                    {
                        "type":"image_url",
                        "image_url":{
                            "url":
                            f"data:image/jpeg;base64,{img_base64}"
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content

_env = my_mujoco_env()
mj_model = _env.model()
mj_data = _env.data()
img = _env.get_rgb()

ans = ask_llm(img)
print(ans)
# while True:
#     img = _env.get_rgb()
#     cv2.imshow("test", img)
#     cv2.waitKey(1)



