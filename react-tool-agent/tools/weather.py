"""
天气查询工具：查询城市天气
"""
import json
from urllib.request import urlopen, Request
from urllib.error import URLError
from tools.base import BaseTool


class WeatherTool(BaseTool):
    name = "get_weather"
    description = "查询指定城市的实时天气信息（温度、湿度、风速等）。当用户询问天气相关问题时使用。"
    parameters = {
        "city": {
            "type": "string",
            "description": "城市名称，如 'Beijing'、'Shanghai'、'Tokyo'",
        },
    }

    def execute(self, city: str) -> str:
        try:
            # 使用 Open-Meteo 免费天气 API（无需 API Key）
            # 先通过 geocoding API 获取城市坐标
            geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1&language=zh"
            req = Request(geo_url, headers={"User-Agent": "ReAct-Agent/1.0"})
            with urlopen(req, timeout=10) as resp:
                geo_data = json.loads(resp.read())

            if not geo_data.get("results"):
                return f"未找到城市 '{city}'，请尝试英文名"

            location = geo_data["results"][0]
            lat, lon = location["latitude"], location["longitude"]
            name = location.get("name", city)
            country = location.get("country", "")

            # 获取天气
            weather_url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&current=temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code"
                f"&timezone=auto"
            )
            req2 = Request(weather_url, headers={"User-Agent": "ReAct-Agent/1.0"})
            with urlopen(req2, timeout=10) as resp2:
                weather_data = json.loads(resp2.read())

            current = weather_data.get("current", {})
            weather_codes = {
                0: "晴天", 1: "大部晴朗", 2: "多云", 3: "阴天",
                45: "雾", 51: "小雨", 61: "中雨", 71: "小雪", 95: "雷暴",
            }
            wcode = current.get("weather_code", 0)
            weather_desc = weather_codes.get(wcode, f"未知({wcode})")

            return (
                f"城市: {name}, {country}\n"
                f"天气: {weather_desc}\n"
                f"温度: {current.get('temperature_2m', '?')}°C\n"
                f"湿度: {current.get('relative_humidity_2m', '?')}%\n"
                f"风速: {current.get('wind_speed_10m', '?')} km/h"
            )
        except URLError:
            return f"[模拟天气] {city}: 晴转多云, 22°C~28°C, 湿度 65%, 风力 3级 (离线演示模式)"
        except Exception as e:
            return f"天气查询失败: {e}"
