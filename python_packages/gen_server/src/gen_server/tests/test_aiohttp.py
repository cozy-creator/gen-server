import json
import unittest
import aiohttp
from aiohttp.test_utils import AioHTTPTestCase
from aiohttp import web
from gen_server.api import get_react_components, get_checkpoints, routes
import os



class GenServerTest(AioHTTPTestCase):

    async def get_application(self):
        """
        Override the get_app method to return your application.
        """
        # initialize_mock()
        app = web.Application()
        app.add_routes(routes)
        return app

    async def asyncSetUp(self):
        self.serverUrl = 'https://p5msj35vzzc9s0-8881.proxy.runpod.net'
        self.testClient = aiohttp.ClientSession(base_url=self.serverUrl)

    async def tearDownAsync(self):
        await self.testClient.close()

    async def test_get_react_components(self):
        res = await get_react_components({})
        new_resp = json.loads(res.text())
        keys = new_resp.keys().__len__()
        self.assertIsInstance(keys, int, f"Expected 'keys' to be INT, got {type(keys)}")

    async def test_get_checkpoints(self):
        res = await get_checkpoints({})
        new_resp = json.loads(res.text())
        keys = new_resp.keys().__len__()
        self.assertIsInstance(keys, int, f"Expected 'keys' to be INT, got {type(keys)}")

    async def test_handle_post(self):
        num_of_images = 2
        data = {
            'models': {
                'break_domain_xl_v05g': num_of_images
            },
            'positive_prompt': 'A white bird',
            'negative_prompt': 'Bird',
            'random_seed': 1,
            'aspect_ratio': '1/1',
        }
        res = await self.testClient.post('/generate', json=data)
        response = await res.read()
        json_strings = response.decode('utf-8').strip().split('\n')
        dict_list = [json.loads(json_str) for json_str in json_strings]

        self.assertEqual(dict_list.__len__(), num_of_images)

        # Print the result
        for item in dict_list:
            # Assert that 'output' exists in the item
            self.assertIn('output', item)
            # Assert that 'url' exists in 'output'
            self.assertIn('url', item['output'])
            # Assert that 'is_temp' exists in 'output'
            self.assertIn('is_temp', item['output'])
            # Additional check: Ensure that 'url' is a non-empty string
            self.assertIsInstance(item['output']['url'], str)
            self.assertTrue(len(item['output']['url']) > 0)
            # Additional check: Ensure that 'is_temp' is a boolean
            self.assertIsInstance(item['output']['is_temp'], bool)

    async def test_upload(self):
        form = aiohttp.FormData()

        # Add the file to the form data
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets/sample.png')
        form.add_field('file',
                       open(img_path, 'rb'),
                       filename='sample.png',
                       content_type='image/png')
        res = await self.testClient.post('/upload', data=form)
        response = await res.json()

        self.assertIn('success', response, "Key 'success' not found in response")
        self.assertIsInstance(response['success'], bool)
        if res.status == 201:
            self.assertTrue(
                response['success'],
                f"Expected 'success' to be True, got {response['success']}"
            )
            self.assertIn('url', response, "Key 'url' not found in response")
        else:
            self.assertFalse(
                response['success'],
                f"Expected 'success' to be False, got {response['success']}"
            )
            self.assertIn('error', response, "Key 'error' not found in response")
            self.assertIsInstance(response['error'], str)

    async def test_media(self):
        res = await self.testClient.get('/media')
        response = await res.json()

        self.assertIn('success', response, "Key 'success' not found in response")
        self.assertIsInstance(response['success'], bool)
        if res.status == 200:
            self.assertTrue(
                response['success'],
                f"Expected 'success' to be True, got {response['success']}"
            )
            self.assertIn('files', response, "Key 'files' not found in response")
            self.assertIsInstance(response['files'], list)
            for item in response['files']:
                self.assertIsInstance(item, str)
        else:
            self.assertFalse(
                response['success'],
                f"Expected 'success' to be False, got {response['success']}"
            )
            self.assertIn('error', response, "Key 'error' not found in response")
            self.assertIsInstance(response['error'], str)


if __name__ == '__main__':
    unittest.main()