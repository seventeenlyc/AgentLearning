import requests

def get_weather(city:str):
    """
    通过调用 wttr.in API 查询真实的天气信息。
    """
    url = f"https://wttr.in/{city}?format=j1"

    try:
        #发起网络请求
        response=requests.get(url)
        #检查是否状态码为200
        response.raise_for_status()
        data=response.json()
        current_condiction=data['current_condition'][0]
        weatherDesc=current_condiction["weatherDesc"][0]["value"]
        temp_C=current_condiction["temp_C"]
        return f"{city}当前天气：{weatherDesc}，当前温度：{temp_C}"
    except requests.exceptions.RequestException as e:
        return f"错误：网络问题{e}"
    except (KeyError, IndexError) as e:
        return f"错误：解析天气数据失败，可能是城市输入错误"




