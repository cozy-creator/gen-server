import aiohttp
import json
from aiohttp import web
import asyncio
from typing import List, Dict, Any, AsyncGenerator


async def generate_urls() -> AsyncGenerator[Dict[str, Any], None]:
    # This is a placeholder function. In a real scenario, this would be
    # your actual image generation function that yields URLs over time.
    urls = [
        {"url": f"http://example.com/image_{i}.jpg", "timestamp": i} 
        for i in range(5)
    ]
    for url in urls:
        await asyncio.sleep(1)  # Simulate some processing time
        yield url


async def handle_post(request: web.Request) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        reason='OK',
        headers={'Content-Type': 'application/json'}
    )
    await response.prepare(request)

    try:
        async for url in generate_urls():
            json_response = json.dumps(url)
            await response.write((json_response + "\n").encode('utf-8'))
    except Exception as e:
        # Handle the exception, you might want to log it or send an error response
        error_message = json.dumps({"error": str(e)})
        await response.write((error_message + "\n").encode('utf-8'))

    await response.write_eof()
    return response


if __name__ == '__main__':
    app = web.Application()
    app.router.add_post('/', handle_post)
    web.run_app(app, host='localhost', port=8080)
