import json
import unittest
from aiohttp.test_utils import AioHTTPTestCase
from aiohttp import web
from gen_server.api import get_react_components, get_checkpoints, handle_post
from gen_server.globals import API_ENDPOINTS


class GenServerTest(AioHTTPTestCase):
    async def get_application(self):
        """
        Override the get_app method to return your application.
        """
        async def hello(request):
            return web.Response(text='Hello, world')

        app = web.Application()
        app.add_routes(API_ENDPOINTS.items())
        print('Endpoints>>', API_ENDPOINTS.items())
        return app
    async def test_get_react_components(self):
        res = await get_react_components({})
        newJ = json.loads(res.text)
        keys = newJ.keys().__len__()
        self.assertEqual(keys, 1)

    async def test_get_checkpoints(self):
        res = await get_checkpoints({})
        self.assertEqual(res.text, '{}')

    # async def test_handle_post(self):
    #     data = {
    #         'model': '',
    #         'positive_prompt': '',
    #         'negative_prompt': '',
    #         'random_seed': '',
    #         'aspect_ratio': '',
    #     }
    #
    #     req = {
    #         'data': data,
    #         'method': 'POST',
    #     }
    #     print(req)
    #     res = await handle_post(req)
    #     print(res)
    #     self.assertEqual('{}', '{}')


if __name__ == '__main__':
    unittest.main()