from fastmcp import Client
from fastmcp.client.client import CallToolResult
from fastmcp.client.transports import StreamableHttpTransport
import asyncio

from mcp_types import Tool


def get_client(url: str) -> Client[StreamableHttpTransport]:
    transport = StreamableHttpTransport(url)
    return Client(transport=transport)

async def main():
    url: str = 'http://localhost:8020/weatherapi-mcp-server'
    client: Client[StreamableHttpTransport] = get_client(url)
    async with client:
        mcp_available_tools: list[Tool] = await client.list_tools()
        for tool in mcp_available_tools:
            if tool.name == 'get_weather_condition':
                arg_location = 'Cairo'
                response: CallToolResult = await client.call_tool(tool.name, {'location': arg_location})
                print(f'Tool: {tool.name}, url={url}')
                print(f'Response::data: {response.data}')



if __name__ == '__main__':
    """Running testing loop"""
    asyncio.run(main=main())

