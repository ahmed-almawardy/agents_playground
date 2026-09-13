import os
from datetime import datetime

from dotenv import load_dotenv
from fastmcp import FastMCP
from aiohttp import ClientSession

load_dotenv(verbose=True)

mcp = FastMCP('mcp-weatherapi')


@mcp.tool(description='MCP Weather API, Getting a Weather condition for a specific location')
async def get_weather_condition(location: str, at_datetime: datetime|None=None) -> dict:
    if not at_datetime: at_datetime = datetime.now()
    url = f"https://api.weatherapi.com/v1/current.json?key={os.environ.get('WEATHERAPI_KEY')}&q={location}&dt={at_datetime}"
    async with ClientSession() as requester:
        response = await requester.get(url)
        try:
            response.raise_for_status()
        except:
            return {'error': "couldn't retrieve forecast for this location, please try again later."}
        return await response.json()
    
if  __name__ == '__main__':
    mcp.run(
        transport='streamable-http',
        host='localhost',
        port='8020',
        path='/weatherapi-mcp-server'
    )